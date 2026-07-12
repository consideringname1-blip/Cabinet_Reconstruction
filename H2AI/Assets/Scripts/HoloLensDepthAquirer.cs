using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections;
using UnityEngine;

public enum DepthSensorType
{
    AHAT,
    LONGTHROW
}

public class HoloLensDepthAquirer : MonoBehaviour
{
    public const string AHATSensorName = "AHAT";
    public const string LongThrowSensorName = "LONGTHROW";

    public const bool EnableAhatUploadGuard = false;
    public const int AHATMinDepthMm = 200;
    public const int AHATMaxReliableDepthMm = 1000;
    public const int AHATMinUsableDepthPixels = 4096;
    public const int AHATMaxUploadPngBytes = 450000;

    [SerializeField] bool _enable_sensor_update = false;
    [SerializeField, Min(1f)] private float maxSensorUpdateHz = 15f;

    // Sensor shown by the local preview; capture paths always select a sensor explicitly.
    [SerializeField] DepthSensorType _depthSensorType = DepthSensorType.AHAT;

    private class DepthSensorState
    {
        public DepthSensorType SensorType;
        public hl2da.SENSOR_ID SensorId;
        public int Width;
        public int Height;
        public int BufferFrames;
        public Texture2D Texture;
        public byte[] RawBuffer;
        public byte[] FlipBuffer;
        public float[,] PoseLatest;
        public int LastFramestamp = -1;
        public bool Initialized;
        public bool HasFrame;
    }

    private DepthSensorState ahatState;
    private DepthSensorState longThrowState;

    private Texture2D tex_grayscale;
    public Texture2D tex_grayscale_publish;
    public float[,] pose_publish;
    public DepthSensorType publishedDepthSensorType { get; private set; } = DepthSensorType.AHAT;
    public string publishedDepthSensorName { get; private set; } = AHATSensorName;

    private readonly Dictionary<hl2da.SENSOR_ID, int> last_framestamp = new Dictionary<hl2da.SENSOR_ID, int>();

    private bool invalidate_depth;
    private bool _depthInitialized;
    private bool ahatStartPending;
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

        // hl2ss opens Long Throw first, waits for it to be stable, then opens AHAT.
        // hl2da exposes depth lock bypass instead; initialize Long Throw first and keep both streams warm.
        hl2da.user.BypassDepthLock_RM(true);

        ahatState = CreateDepthSensorState(DepthSensorType.AHAT);
        longThrowState = CreateDepthSensorState(DepthSensorType.LONGTHROW);
        InitializeSensor(longThrowState);
        ahatStartPending = true;

        tex_grayscale = GetState(_depthSensorType)?.Texture;
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
            UpdateFrame(DepthSensorType.LONGTHROW);
            StartAhatAfterLongThrowFirstFrame();
            UpdateFrame(DepthSensorType.AHAT);
#endif
        }
    }

    private void StartAhatAfterLongThrowFirstFrame()
    {
        if (!ahatStartPending)
        {
            return;
        }
        if (longThrowState == null || !longThrowState.HasFrame || ahatState == null)
        {
            return;
        }

        InitializeSensor(ahatState);
        ahatStartPending = false;
        if (_depthSensorType == DepthSensorType.AHAT)
        {
            tex_grayscale = ahatState.Texture;
        }
    }

    private DepthSensorState CreateDepthSensorState(DepthSensorType sensorType)
    {
        if (sensorType == DepthSensorType.AHAT)
        {
            return new DepthSensorState
            {
                SensorType = sensorType,
                SensorId = hl2da.SENSOR_ID.RM_DEPTH_AHAT,
                Width = 512,
                Height = 512,
                BufferFrames = 90,
            };
        }

        return new DepthSensorState
        {
            SensorType = sensorType,
            SensorId = hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW,
            Width = 320,
            Height = 288,
            BufferFrames = 15,
        };
    }

    private void InitializeSensor(DepthSensorState state)
    {
#if WINDOWS_UWP
        hl2da.user.Initialize(state.SensorId, state.BufferFrames);
        hl2da.user.SetEnable(state.SensorId, true);
#endif
        state.Texture = new Texture2D(state.Width, state.Height, TextureFormat.R16, false);
        state.RawBuffer = new byte[state.Width * state.Height * 2];
        state.FlipBuffer = new byte[state.Width * state.Height * 2];
        state.PoseLatest = new float[hl2da.user.POSE_ROWS, hl2da.user.POSE_COLS];
        state.Initialized = true;
    }

    private DepthSensorState GetState(DepthSensorType sensorType)
    {
        return sensorType == DepthSensorType.AHAT ? ahatState : longThrowState;
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

    private void CountFrozenDepthTexture(Texture2D texture)
    {
        NativeArray<byte> rawBytes = texture.GetRawTextureData<byte>();
        for (int i = 0; i + 1 < rawBytes.Length; i += 2)
        {
            ushort depthMm = (ushort)(rawBytes[i] | (rawBytes[i + 1] << 8));
            if (depthMm != 0)
            {
                lastFrozenDepthRawNonzeroPixels++;
                lastFrozenDepthValidPixels++;
            }
        }
    }

    private void SanitizeFrozenDepthTexture(Texture2D texture, DepthSensorType sensorType)
    {
        ResetFrozenDepthStats();

        if (sensorType != DepthSensorType.AHAT)
        {
            CountFrozenDepthTexture(texture);
            return;
        }

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

    public string GetDepthSensorName(DepthSensorType sensorType)
    {
        return sensorType == DepthSensorType.AHAT ? AHATSensorName : LongThrowSensorName;
    }

    public string GetPublishedDepthSensorName()
    {
        return publishedDepthSensorName;
    }

    public bool IsUsingAhatSensor()
    {
        return _depthSensorType == DepthSensorType.AHAT;
    }

    public bool IsPublishedAhatSensor()
    {
        return publishedDepthSensorType == DepthSensorType.AHAT;
    }

    public bool IsFrozenAhatDepthUsable()
    {
        return IsPublishedAhatSensor() && lastFrozenAhatDepthUsable;
    }

    void UpdateFrame(DepthSensorType sensorType)
    {
        DepthSensorState state = GetState(sensorType);
        if (state == null || !state.Initialized)
        {
            return;
        }

        ulong fbRefTimestamp;
        using (hl2da.framebuffer fbRef = hl2da.framebuffer.GetFrame(state.SensorId, -2))
        {
            if (fbRef.Status != hl2da.STATUS.OK) { return; }
            fbRefTimestamp = fbRef.Timestamp;
            UpdateSensorData(state, fbRef);
        }

        using (hl2da.framebuffer fb = hl2da.framebuffer.GetFrame(state.SensorId, fbRefTimestamp, hl2da.TIME_PREFERENCE.NEAREST, false))
        {
            if (fb.Status == hl2da.STATUS.OK) { UpdateSensorData(state, fb); }
        }
    }

    void UpdateSensorData(DepthSensorState state, hl2da.framebuffer fb)
    {
        if (fb.Framestamp <= state.LastFramestamp) { return; }
        state.LastFramestamp = fb.Framestamp;
        last_framestamp[fb.Id] = fb.Framestamp;

        switch (fb.Id)
        {
            case hl2da.SENSOR_ID.RM_DEPTH_AHAT: UpdateRMDepthAHAT(state, fb); break;
            case hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW: UpdateRMDepthLongthrow(state, fb); break;
        }
    }

    void UpdateRMDepthAHAT(DepthSensorState state, hl2da.framebuffer fb)
    {
        if (invalidate_depth) { hl2da.IMT_ZHTInvalidate(fb.Buffer(0), fb.Buffer(0)); }
        UpdateDepthTextureAndPose(state, fb);
    }

    void UpdateRMDepthLongthrow(DepthSensorState state, hl2da.framebuffer fb)
    {
        if (invalidate_depth) { hl2da.IMT_ZLTInvalidate(fb.Buffer(2), fb.Buffer(0), fb.Buffer(0)); }
        UpdateDepthTextureAndPose(state, fb);
    }

    void UpdateDepthTextureAndPose(DepthSensorState state, hl2da.framebuffer fb)
    {
        int byteCount = fb.Length(0) * sizeof(ushort);
        int maxByteCount = Mathf.Min(byteCount, state.RawBuffer.Length);

        Marshal.Copy(fb.Buffer(0), state.RawBuffer, 0, maxByteCount);
        FlipRawBufferVerticallyR16(state.RawBuffer, state.FlipBuffer, state.Width, state.Height);

        state.Texture.LoadRawTextureData(state.FlipBuffer);
        state.Texture.Apply(false);

        hl2da.user.Copy<float>(fb.Buffer(3), state.PoseLatest, state.PoseLatest.Length);
        state.HasFrame = true;

        if (state.SensorType == _depthSensorType)
        {
            tex_grayscale = state.Texture;
        }
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

    public bool FreezeCurrentAhatFrame()
    {
        return FreezeCurrentFrame(DepthSensorType.AHAT);
    }

    public bool FreezeCurrentLongThrowFrame()
    {
        return FreezeCurrentFrame(DepthSensorType.LONGTHROW);
    }

    public bool FreezeCurrentFrame(DepthSensorType sensorType)
    {
        DepthSensorState state = GetState(sensorType);
        string sensorName = GetDepthSensorName(sensorType);

        if (state == null || state.Texture == null)
        {
            ResetFrozenDepthStats();
            Game_M.initialize.XianShi("dp_freeze_ERR_tex_null_" + sensorName);
            return false;
        }

        if (!state.HasFrame)
        {
            ResetFrozenDepthStats();
            Game_M.initialize.XianShi("dp_freeze_ERR_no_frame_" + sensorName);
            return false;
        }

        if (state.PoseLatest == null)
        {
            ResetFrozenDepthStats();
            Game_M.initialize.XianShi("dp_freeze_ERR_pose_null_" + sensorName);
            return false;
        }

        if (tex_grayscale_publish == null ||
            tex_grayscale_publish.width != state.Texture.width ||
            tex_grayscale_publish.height != state.Texture.height ||
            tex_grayscale_publish.format != state.Texture.format)
        {
            tex_grayscale_publish = new Texture2D(
                state.Texture.width,
                state.Texture.height,
                state.Texture.format,
                false
            );
            Game_M.initialize.XianShi("dp_freeze_01_create_tex_" + sensorName);
        }

        tex_grayscale_publish.LoadRawTextureData(state.Texture.GetRawTextureData());
        SanitizeFrozenDepthTexture(tex_grayscale_publish, sensorType);
        tex_grayscale_publish.Apply(false);

        pose_publish = CloneFloat2D(state.PoseLatest);
        _depthSensorType = sensorType;
        tex_grayscale = state.Texture;
        publishedDepthSensorType = sensorType;
        publishedDepthSensorName = sensorName;

        if (sensorType != DepthSensorType.AHAT || lastFrozenAhatDepthUsable || !EnableAhatUploadGuard)
        {
            Game_M.initialize.XianShi("dp_freeze_02_done_" + sensorName);
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
            hl2da.user.SetEnable(hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW, false);
            hl2da.user.SetEnable(hl2da.SENSOR_ID.RM_DEPTH_AHAT, false);
            _depthInitialized = false;
        }
#endif
        ahatStartPending = false;
        ReleaseSensorState(ahatState);
        ReleaseSensorState(longThrowState);
        ahatState = null;
        longThrowState = null;
        tex_grayscale = null;

        if (tex_grayscale_publish != null)
        {
            Destroy(tex_grayscale_publish);
            tex_grayscale_publish = null;
        }
    }

    private void ReleaseSensorState(DepthSensorState state)
    {
        if (state == null)
        {
            return;
        }

        if (state.Texture != null)
        {
            Destroy(state.Texture);
            state.Texture = null;
        }

        state.RawBuffer = null;
        state.FlipBuffer = null;
        state.PoseLatest = null;
        state.Initialized = false;
        state.HasFrame = false;
    }
}
