using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TMPro;
using UnityEngine;

enum DepthSensorType
{
    AHAT,
    LONGTHROW
}

public class HoloLensDepthAquirer : MonoBehaviour
{
    [SerializeField] bool _enable_sensor_update = false;

    // set Depth Sensor Mode (default All)
    [SerializeField] DepthSensorType _depthSensorType = DepthSensorType.AHAT;

    [SerializeField] bool _enablePreview = false;
    //[SerializeField] HoloLensAquiredDepthPreviewController _previewController;

    public ShuJuQingQiu _publisher;

    //public Texture colormap_texture;
    //public Shader colormap_shader;
    //public Shader grayscale_shader;

    private Texture2D tex_grayscale;

    private Texture2D tex_grayscale_publish;
    private byte[] publish_flip_buffer_depth;

    //private RenderTexture tex_color;

    //private Material colormap_mat;
    //private Material grayscale_mat;

    private Dictionary<hl2da.SENSOR_ID, string> sensor_names;
    private Dictionary<hl2da.SENSOR_ID, int> last_framestamp;

    private bool invalidate_depth;

    private Dictionary<hl2da.SENSOR_ID, float[,]> rm_uv2xy = new Dictionary<hl2da.SENSOR_ID, float[,]>();
    private Dictionary<hl2da.SENSOR_ID, float[,]> rm_mapxy = new Dictionary<hl2da.SENSOR_ID, float[,]>();
    private Dictionary<hl2da.SENSOR_ID, float[]> rm_intrinsics = new Dictionary<hl2da.SENSOR_ID, float[]>();

    // Start is called before the first frame update
    void Start()
    {
        // Set invalid depth pixels to zero
        invalidate_depth = true;

#if WINDOWS_UWP
        hl2da.user.InitializeComponents();
        hl2da.user.OverrideWorldCoordinateSystem(); // Link Unity and plugin coordinate systems

        Initialize_Dictionaries();

        hl2da.user.BypassDepthLock_RM(true); // Allows simultaneous access to AHAT and longthrow depth

        Update_Calibration();

        // Use a buffer size of 2 seconds (except longthrow and PV)
        if (_depthSensorType == DepthSensorType.AHAT)
        {
            hl2da.user.Initialize(hl2da.SENSOR_ID.RM_DEPTH_AHAT, 90); // Buffer size limited by memory // 45 Hz
            hl2da.user.SetEnable(hl2da.SENSOR_ID.RM_DEPTH_AHAT, true);

            tex_grayscale = new Texture2D(512, 512, TextureFormat.R16, false);

            tex_grayscale_publish = new Texture2D(512, 512, TextureFormat.R16, false);
            publish_flip_buffer_depth = new byte[512 * 512 * 2];
            //tex_color = new RenderTexture(512, 512, 0, RenderTextureFormat.BGRA32);

            //if (_enablePreview)
            //{
            //    _previewController.InitializePreview_AHAT(tex_grayscale, tex_color);
            //}

            //colormap_mat = new Material(colormap_shader);
            //colormap_mat.SetTexture("_ColorMapTex", colormap_texture);
            //colormap_mat.SetFloat("_Lf", 0.0f / 65535.0f);
            //colormap_mat.SetFloat("_Rf", 1055.0f / 65535.0f);
        }
        else if (_depthSensorType == DepthSensorType.LONGTHROW) 
        {
            hl2da.user.Initialize(hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW, 15); // Buffer size limited by internal buffer - Maximum is 18 // 5 Hz
            hl2da.user.SetEnable(hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW, true);

            tex_grayscale = new Texture2D(320, 288, TextureFormat.R16, false);
            //tex_color = new RenderTexture(320, 288, 0, RenderTextureFormat.BGRA32);

            //if (_enablePreview)
            //{
            //    _previewController.InitializePreview_LongThrow(tex_grayscale, tex_color);
            //}

            //colormap_mat = new Material(colormap_shader);
            //colormap_mat.SetTexture("_ColorMapTex", colormap_texture);
            //colormap_mat.SetFloat("_Lf", 0.0f / 65535.0f);
            //colormap_mat.SetFloat("_Rf", 3000.0f / 65535.0f);
        }

        //grayscale_mat = new Material(grayscale_shader);
#endif
        _enable_sensor_update = true;
    }

    // Update is called once per frame
    void Update()
    {
        if (_enable_sensor_update)
        {
#if WINDOWS_UWP
            Update_Frame();
#endif
        }
    }

    public void Switch_DepthUpdate()
    {
        if (_enable_sensor_update) { _enable_sensor_update = false; }
        else { _enable_sensor_update = true; }
    }

    private void FlipTextureVerticallyR16(Texture2D src, Texture2D dst, byte[] flipBuffer)
    {
        int width = src.width;
        int height = src.height;
        int rowBytes = width * 2;

        var srcRaw = src.GetRawTextureData<byte>();

        for (int y = 0; y < height; y++)
        {
            int srcOffset = y * rowBytes;
            int dstOffset = (height - 1 - y) * rowBytes;

            for (int i = 0; i < rowBytes; i++)
            {
                flipBuffer[dstOffset + i] = srcRaw[srcOffset + i];
            }
        }

        dst.LoadRawTextureData(flipBuffer);
        dst.Apply(false);
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

    void Initialize_Dictionaries()
    {
        sensor_names = new Dictionary<hl2da.SENSOR_ID, string>();
        last_framestamp = new Dictionary<hl2da.SENSOR_ID, int>();

        for (hl2da.SENSOR_ID id = hl2da.SENSOR_ID.RM_VLC_LEFTFRONT; id <= hl2da.SENSOR_ID.EXTENDED_VIDEO; ++id)
        {
            sensor_names[id] = id.ToString();
            last_framestamp[id] = -1;
        }
    }

    string CentralPoints(hl2da.SENSOR_ID id)
    {
        float[,] image_points = new float[1, 2];
        float[,] camera_points = new float[1, 2];

        switch (id)
        {
            case hl2da.SENSOR_ID.RM_DEPTH_AHAT: image_points[0, 0] = 256.0f; image_points[0, 1] = 256.0f; break;
            case hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW: image_points[0, 0] = 160.0f; image_points[0, 1] = 144.0f; break;
            default: return "";
        }

        camera_points[0, 0] = 0.0f;
        camera_points[0, 1] = 0.0f;

        float[,] mpoint = hl2da.user.RM_MapImagePointToCameraUnitPlane(id, image_points); // get image center in camera space
        float[,] ppoint = hl2da.user.RM_MapCameraSpaceToImagePoint(id, camera_points); // get image principal point

        return string.Format(" c'=[{0}, {1}], c=[{2}, {3}]", mpoint[0, 0], mpoint[0, 1], ppoint[0, 0], ppoint[0, 1]);
    }

    string Calibration(hl2da.SENSOR_ID id)
    {
        switch (id)
        {
            case hl2da.SENSOR_ID.RM_VLC_LEFTFRONT:
            case hl2da.SENSOR_ID.RM_VLC_LEFTLEFT:
            case hl2da.SENSOR_ID.RM_VLC_RIGHTFRONT:
            case hl2da.SENSOR_ID.RM_VLC_RIGHTRIGHT:
            case hl2da.SENSOR_ID.RM_DEPTH_AHAT:
            case hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW: break;
            default: return "";
        }

        hl2da.user.RM_GetIntrinsics(id, out float[,] uv2xy, out float[,] mapxy, out float[] k);

        rm_uv2xy[id] = uv2xy;
        rm_mapxy[id] = mapxy;
        rm_intrinsics[id] = k;

        return string.Format(" fx={0}, fy={1}, cx={2}, cy={3}", k[0], k[1], k[2], k[3]);
    }

    void Update_Calibration()
    {
        string text = "";

        //for (hl2da.SENSOR_ID id = hl2da.SENSOR_ID.RM_DEPTH_AHAT; id <= hl2da.SENSOR_ID.RM_IMU_GYROSCOPE; ++id)
        //{
        //    float[,] extrinsics = hl2da.user.RM_GetExtrinsics(id);
        //    string text = sensor_names[id] + " Calibration: extrinsics=" + PoseToString(extrinsics) + CentralPoints(id) + Calibration(id);
        //    calibrations[(int)id].GetComponent<TextMeshPro>().text = text;
        //}
        if (_depthSensorType == DepthSensorType.AHAT)
        {
            float[,] extrinsics = hl2da.user.RM_GetExtrinsics(hl2da.SENSOR_ID.RM_DEPTH_AHAT);
            text += sensor_names[hl2da.SENSOR_ID.RM_DEPTH_AHAT] + " Calibration: extrinsics=" + PoseToString(extrinsics) + CentralPoints(hl2da.SENSOR_ID.RM_DEPTH_AHAT) + Calibration(hl2da.SENSOR_ID.RM_DEPTH_AHAT) + "\n";
        }

        if (_depthSensorType == DepthSensorType.LONGTHROW)
        {
            float[,] extrinsics = hl2da.user.RM_GetExtrinsics(hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW);
            text += sensor_names[hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW] + " Calibration: extrinsics=" + PoseToString(extrinsics) + CentralPoints(hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW) + Calibration(hl2da.SENSOR_ID.RM_DEPTH_LONGTHROW) + "\n";
        }

        //_previewController.PreviewCalibration(text);
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

        // Load frame data into textures
        tex_grayscale.LoadRawTextureData(fb.Buffer(0), fb.Length(0) * sizeof(ushort));  // Depth is u16
        //tex_grayscale.LoadRawTextureData(fb.Buffer(0), fb.Length(0));  // Depth is u16
        tex_grayscale.Apply();

        // encode image to png
        //byte[] frameData = ImageConversion.EncodeToPNG(tex_grayscale);

        //Graphics.Blit(tex_grayscale, tex_color, colormap_mat); // Apply color map to Depth

        // Display pose
        float[,] pose = hl2da.user.Unpack2D<float>(fb.Buffer(3), hl2da.user.POSE_ROWS, hl2da.user.POSE_COLS);
        //tmp_ht_pose.text = sensor_names[fb.Id] + " Pose: " + PoseToString(pose);

        //byte[] frameData = new byte[fb.Length(0) * sizeof(ushort) * 2];
        //Marshal.Copy(fb.Buffer(0), frameData, 0, fb.Length(0) * sizeof(ushort) * 2);

        //_previewController.UpdateFPSText();
        Publish(tex_grayscale, pose);
    }

    void Update_RM_Depth_Longthrow(hl2da.framebuffer fb)
    {
        if (invalidate_depth) { hl2da.IMT_ZLTInvalidate(fb.Buffer(2), fb.Buffer(0), fb.Buffer(0)); }

        // Load frame data into textures
        tex_grayscale.LoadRawTextureData(fb.Buffer(0), fb.Length(0) * sizeof(ushort)); // Depth is u16
        tex_grayscale.Apply();
        //Graphics.Blit(tex_grayscale, tex_color, colormap_mat); // Apply color map to Depth

        // Display pose
        float[,] pose = hl2da.user.Unpack2D<float>(fb.Buffer(3), hl2da.user.POSE_ROWS, hl2da.user.POSE_COLS);
        //tmp_lt_pose.text = sensor_names[fb.Id] + " Pose: " + PoseToString(pose);
    }

    string PoseToString(float[,] pose)
    {
        return string.Format("[[{0}, {1}, {2}, {3}], [{4}, {5}, {6}, {7}], [{8}, {9}, {10}, {11}], [{12}, {13}, {14}, {15}]]", pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3], pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3], pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3], pose[3, 0], pose[3, 1], pose[3, 2], pose[3, 3]);
    }

    //void Publish(byte[] image, float[,] pose)
    //{
    //    //_publisher.PublishMessage(image, pose);
    //}

    public bool PublishStatus = true;

    void Publish(Texture2D image, float[,] pose)
    {
        if (!PublishStatus) return;
        FlipTextureVerticallyR16(image, tex_grayscale_publish, publish_flip_buffer_depth);
        _publisher.PublishDPMessage(image, pose);

    }
}
