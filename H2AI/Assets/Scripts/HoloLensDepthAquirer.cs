using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections;
using UnityEngine;

enum DepthSensorType
{
    AHAT,
    LONGTHROW
}

public class HoloLensDepthAquirer : MonoBehaviour
{
    public const bool EnableAhatUploadGuard = false;
    public const int AHATMinDepthMm = 200;
    public const int AHATMaxReliableDepthMm = 1000;
    public const int AHATMinUsableDepthPixels = 4096;
    public const int AHATMaxUploadPngBytes = 450000;

    [SerializeField] bool _enable_sensor_update = false;
    [SerializeField, Min(1f)] private float maxSensorUpdateHz = 15f;

    // Keep LONGTHROW support for remote-depth capture even though model upload uses AHAT.
    [SerializeField] DepthSensorType _depthSensorType = DepthSensorType.AHAT;

    private Texture2D tex_grayscale;
    private byte[] depth_raw_buffer;
    private byte[] depth_flip_buffer;

    private float[,] pose_latest;
    public Texture2D tex_grayscale_publish;
    public float[,] pose_publish;

    private readonly Dictionary<hl2da.SENSOR_ID, int> last_framestamp = new Dictionary<hl2da.SENSOR_ID, int>();

    private bool invalidate_depth;

    private bool _depthInitialized;
    private float nextSensorUpdateTime;
    public int lastFrozenDepthRawNonzeroPixels { get; private set; }
    public int lastFrozenDepthValidPixels { get; private set; }
    public int lastFrozenDepthClippedPixels { get; private set; }
    public bool lastFrozenAhatDepthUsable { get; private set; }

    void Start()
    {
        invalidate_depth = true;
        last_framestamp[hl2da.SENSOR_ID.RM_DEPTH_AHAT] = -1;
        last_framestamp[hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW] = -1;

#if WINDOWS_UWP
        hl2da.user.InitializeComponents();
        hl2da.user.OverrideWorldCoordinateSystem(); // Link Unity and plugin coordinate systems

        hl2da.user.BypassDepthLock_RM(true); // Allows simultaneous access to AHAT and longthrow depth

        // Use a buffer size of 2 seconds (except longthrow and PV)
        if (_depthSensorType == DepthSensorType.AHAT)
        {
            hl2da.user.Initialize(hl2da.SENSOR_ID.RM_DEPTH_AHAT, 90); // Buffer size limited by memory // 45 Hz
            hl2da.user.SetEnable(hl2da.SENSOR_ID.RM_DEPTH_AHAT, true);

            tex_grayscale = new Texture2D(512, 512, TextureFormat.R16, false);

            depth_raw_buffer = new byte[512 * 512 * 2];
            depth_flip_buffer = new byte[512 * 512 * 2];
        }
        else if (_depthSensorType == DepthSensorType.LONGTHROW) 
        {
            hl2da.user.Initialize(hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW, 15); // Buffer size limited by internal buffer - Maximum is 18 // 5 Hz
            hl2da.user.SetEnable(hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW, true);

            tex_grayscale = new Texture2D(320, 288, TextureFormat.R16, false);

            depth_raw_buffer = new byte[320 * 288 * 2];
            depth_flip_buffer = new byte[320 * 288 * 2];
        }

        pose_latest = new float[hl2da.user.POSE_ROWS, hl2da.user.POSE_COLS];
        _depthInitialized = true;
#endif
        _enable_sensor_update = true;
    }

    void Update()
    {
        if (_enable_sensor_update && Time.unscaledTime >= nextSensorUpdateTime)
        {
            nextSensorUpdateTime = Time.unscaledTime + (1f / Mathf.Max(1f, maxSensorUpdateHz));
#if WINDOWS_UWP
            Update_Frame();
#endif
        }
    }

    private void FlipRawBufferVerticallyR16(byte[] src, byte[] dst, int width, int height)
    {
        int rowBytes = width * 2; // R16 = 2 bytes per pixel

        for (int y = 0; y < height; y++)
        {
            int srcOffset = y * rowBytes;
            int dstOffset = (height - 1 - y) * rowBytes;

            System.Buffer.BlockCopy(src, srcOffset, dst, dstOffset, rowBytes);
        }
    }

    private void ResetFrozenDepthStats()
    {
        lastFrozenDepthRawNonzeroPixels = 0;
        lastFrozenDepthValidPixels = 0;
        lastFrozenDepthClippedPixels = 0;
        lastFrozenAhatDepthUsable = false;
    }

    private void SanitizeFrozenAhatDepthTexture(Texture2D texture)
    {
        ResetFrozenDepthStats();

        NativeArray<byte> rawBytes = texture.GetRawTextureData<byte>();
        for (int i = 0; i + 1 < rawBytes.Length; i += 2)
        {
            ushort depthMm = (ushort)(rawBytes[i] | (rawBytes[i + 1] << 8));
            if (depthMm == 0)
            {
                continue;
            }

            lastFrozenDepthRawNonzeroPixels++;
            if (depthMm >= AHATMinDepthMm && depthMm <= AHATMaxReliableDepthMm)
            {
                lastFrozenDepthValidPixels++;
                continue;
            }

            rawBytes[i] = 0;
            rawBytes[i + 1] = 0;
            lastFrozenDepthClippedPixels++;
        }

        lastFrozenAhatDepthUsable = lastFrozenDepthValidPixels >= AHATMinUsableDepthPixels;
    }

    public bool IsUsingAhatSensor()
    {
        return _depthSensorType == DepthSensorType.AHAT;
    }

    public bool IsFrozenAhatDepthUsable()
    {
        return IsUsingAhatSensor() && lastFrozenAhatDepthUsable;
    }

    /// <summary>
    /// Aquire new Frame Data from sensor
    /// </summary>
    void Update_Frame()
    {
        if (_depthSensorType == DepthSensorType.AHAT) 
        {
            ulong fb_ref_timestamp;
            using (hl2da.framebuffer fb_ref = hl2da.framebuffer.GetFrame(hl2da.SENSOR_ID.RM_DEPTH_AHAT, -2)) // Use a small delay to allow receiving the optimal frame from the second stream
            {
                if (fb_ref.Status != hl2da.STATUS.OK) { return; }
                fb_ref_timestamp = fb_ref.Timestamp;
                Update_Sensor_Data(fb_ref);
            }

            // Associate frames
            // If no frame matches fb_ref_timestamp exactly then:
            //   hl2da.TIME_PREFERENCE.PAST:    select nearest frame with Timestamp < fb_ref_timestamp
            //   hl2da.TIME_PREFERENCE.NEAREST: select nearest frame, in case of a tie choose Timestamp > fb_ref_timestamp if tiebreak_right=true else choose Timestamp < fb_ref_timestamp
            //   hl2da.TIME_PREFERENCE.FUTURE:  select nearest frame with Timestamp > fb_ref_timestamp
            using (hl2da.framebuffer fb = hl2da.framebuffer.GetFrame(hl2da.SENSOR_ID.RM_DEPTH_AHAT, fb_ref_timestamp, hl2da.TIME_PREFERENCE.NEAREST, false))
            {
                if (fb.Status == hl2da.STATUS.OK) { Update_Sensor_Data(fb); }
            }
        }
        else if (_depthSensorType == DepthSensorType.LONGTHROW)
        {
            ulong fb_ref_timestamp;
            using (hl2da.framebuffer fb_ref = hl2da.framebuffer.GetFrame(hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW, -2)) // Use a small delay to allow receiving the optimal frame from the second stream
            {
                if (fb_ref.Status != hl2da.STATUS.OK) { return; }
                fb_ref_timestamp = fb_ref.Timestamp;
                Update_Sensor_Data(fb_ref);
            }

            // Associate frames
            // If no frame matches fb_ref_timestamp exactly then:
            //   hl2da.TIME_PREFERENCE.PAST:    select nearest frame with Timestamp < fb_ref_timestamp
            //   hl2da.TIME_PREFERENCE.NEAREST: select nearest frame, in case of a tie choose Timestamp > fb_ref_timestamp if tiebreak_right=true else choose Timestamp < fb_ref_timestamp
            //   hl2da.TIME_PREFERENCE.FUTURE:  select nearest frame with Timestamp > fb_ref_timestamp
            using (hl2da.framebuffer fb = hl2da.framebuffer.GetFrame(hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW, fb_ref_timestamp, hl2da.TIME_PREFERENCE.NEAREST, false))
            {
                if (fb.Status == hl2da.STATUS.OK) { Update_Sensor_Data(fb); }
                
            }
        }
    }

    void Update_Sensor_Data(hl2da.framebuffer fb)
    {
        if (fb.Framestamp <= last_framestamp[fb.Id]) { return; } // Repeated frame, nothing to do...
        last_framestamp[fb.Id] = fb.Framestamp;

        switch (fb.Id)
        {
            case hl2da.SENSOR_ID.RM_DEPTH_AHAT: Update_RM_Depth_AHAT(fb); break;
            case hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW: Update_RM_Depth_Longthrow(fb); break;
        }
    }

    void Update_RM_Depth_AHAT(hl2da.framebuffer fb)
    {
        if (invalidate_depth) { hl2da.IMT_ZHTInvalidate(fb.Buffer(0), fb.Buffer(0)); }

        int byteCount = fb.Length(0) * sizeof(ushort);

        Marshal.Copy(fb.Buffer(0), depth_raw_buffer, 0, byteCount);

        FlipRawBufferVerticallyR16(depth_raw_buffer, depth_flip_buffer, tex_grayscale.width, tex_grayscale.height);

        tex_grayscale.LoadRawTextureData(depth_flip_buffer);
        tex_grayscale.Apply(false);

        hl2da.user.Copy<float>(fb.Buffer(3), pose_latest, pose_latest.Length);
    }

    void Update_RM_Depth_Longthrow(hl2da.framebuffer fb)
    {
        if (invalidate_depth) { hl2da.IMT_ZLTInvalidate(fb.Buffer(2), fb.Buffer(0), fb.Buffer(0)); }

        int byteCount = fb.Length(0) * sizeof(ushort);

        Marshal.Copy(fb.Buffer(0), depth_raw_buffer, 0, byteCount);

        FlipRawBufferVerticallyR16(depth_raw_buffer, depth_flip_buffer, tex_grayscale.width, tex_grayscale.height);

        tex_grayscale.LoadRawTextureData(depth_flip_buffer);
        tex_grayscale.Apply(false);

        hl2da.user.Copy<float>(fb.Buffer(3), pose_latest, pose_latest.Length);
    }

    private float[,] CloneFloat2D(float[,] src)
    {
        if (src == null) return null;

        int rows = src.GetLength(0);
        int cols = src.GetLength(1);
        float[,] dst = new float[rows, cols];

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                dst[r, c] = src[r, c];
            }
        }

        return dst;
    }

    public bool FreezeCurrentFrame()
    {
        if (!IsUsingAhatSensor())
        {
            ResetFrozenDepthStats();
            Game_M.initialize.XianShi("dp_freeze_ERR_sensor_not_ahat");
            return false;
        }

        if (tex_grayscale == null)
        {
            Game_M.initialize.XianShi("dp_freeze_ERR_tex_null");
            return false;
        }

        if (pose_latest == null)
        {
            Game_M.initialize.XianShi("dp_freeze_ERR_pose_null");
            return false;
        }

        if (tex_grayscale_publish == null ||
            tex_grayscale_publish.width != tex_grayscale.width ||
            tex_grayscale_publish.height != tex_grayscale.height ||
            tex_grayscale_publish.format != tex_grayscale.format)
        {
            tex_grayscale_publish = new Texture2D(
                tex_grayscale.width,
                tex_grayscale.height,
                tex_grayscale.format,
                false
            );
            Game_M.initialize.XianShi("dp_freeze_01_create_tex");
        }

        tex_grayscale_publish.LoadRawTextureData(tex_grayscale.GetRawTextureData());
        SanitizeFrozenAhatDepthTexture(tex_grayscale_publish);
        tex_grayscale_publish.Apply(false);

        pose_publish = CloneFloat2D(pose_latest);

        if (lastFrozenAhatDepthUsable || !EnableAhatUploadGuard)
        {
            Game_M.initialize.XianShi("dp_freeze_02_done");
        }
        else
        {
            Game_M.initialize.XianShi("dp_freeze_ahat_move_closer");
        }
        return true;
    }

    void OnDestroy()
    {
        ReleaseDepthResources();
    }

    void OnApplicationQuit()
    {
        ReleaseDepthResources();
    }

    void ReleaseDepthResources()
    {
#if WINDOWS_UWP
        if (_depthInitialized)
        {
            if (_depthSensorType == DepthSensorType.AHAT)
            {
                hl2da.user.SetEnable(hl2da.SENSOR_ID.RM_DEPTH_AHAT, false);
            }
            else if (_depthSensorType == DepthSensorType.LONGTHROW)
            {
                hl2da.user.SetEnable(hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW, false);
            }

            _depthInitialized = false;
        }
#endif

        if (tex_grayscale_publish != null)
        {
            Destroy(tex_grayscale_publish);
            tex_grayscale_publish = null;
        }

        if (tex_grayscale != null)
        {
            Destroy(tex_grayscale);
            tex_grayscale = null;
        }
    }
}
