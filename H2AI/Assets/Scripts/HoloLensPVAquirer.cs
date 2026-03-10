using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine.UI;

public class HoloLensPVAquirer : MonoBehaviour
{
    [SerializeField] bool _enable_sensor_update = false;

    //public GameObject pv_image;

    public RawImage pv_image;

    public ushort pv_width = 640;
    public ushort pv_height = 360;
    public byte pv_fps = 30;

    public Texture2D tex_pv_frozen;
    public float[,] pose_pv_frozen;
    public float[,] k_pv_frozen;
    public ushort width_pv_frozen;
    public ushort height_pv_frozen;
    private float[,] pose_latest;
    private float[,] k_latest;

    private hl2da.pv_captureformat pvcf;

    private Texture2D tex_pv;
    private byte[] pv_raw_buffer;
    private byte[] pv_flip_buffer;

    //public HoloLensPVPublisher _publisher;
    public ShuJuQingQiu _publisher;

    // Start is called before the first frame update
    void Start()
    {
#if WINDOWS_UWP
        hl2da.user.InitializeComponents();
        hl2da.user.OverrideWorldCoordinateSystem();
        pvcf = hl2da.user.CreateFormat_PV(pv_width, pv_height, pv_fps, false, false);
        hl2da.user.SetFormat_PV(pvcf);

        hl2da.user.Initialize(hl2da.SENSOR_ID.PV, 15); // Max 18
        hl2da.user.SetEnable(hl2da.SENSOR_ID.PV, true);

        tex_pv = new Texture2D(pvcf.width, pvcf.height, TextureFormat.BGRA32, false);
        pv_raw_buffer = new byte[pvcf.width * pvcf.height * 4];
        pv_flip_buffer = new byte[pvcf.width * pvcf.height * 4];
        pv_image.texture = tex_pv;
#endif
        _enable_sensor_update = true;////
    }

    // Update is called once per frame
    void Update()
    {
        if (_enable_sensor_update)
        {
#if WINDOWS_UWP
            UpdateFrame();
#endif
        }
    }

    public void Switch_PVUpdate()
    {
        if (_enable_sensor_update) { _enable_sensor_update = false; }
        else { _enable_sensor_update = true; }
    }

    void FlipVertical(byte[] src, byte[] dst, int width, int height)
    {
        int rowBytes = width * 4;

        for (int y = 0; y < height; y++)
        {
            int srcOffset = y * rowBytes;
            int dstOffset = (height - 1 - y) * rowBytes;

            System.Buffer.BlockCopy(src, srcOffset, dst, dstOffset, rowBytes);
        }
    }


    void UpdateFrame()
    {
        using var fb = hl2da.framebuffer.GetFrame(hl2da.SENSOR_ID.PV, -1);
        if (fb.Status != hl2da.STATUS.OK) { return; }

        uint stride = hl2da.converter.GetStride_PV(pvcf.width);

        hl2da.converter fc = hl2da.converter.Convert(fb.Buffer(0), stride, pvcf.height, hl2da.IMT_Format.Nv12, hl2da.IMT_Format.Bgra8);

        if (stride != pvcf.width)
        {
            byte[,,] image = hl2da.coprocessor.Crop<byte>(
                fc.Buffer,
                (int)stride,
                pvcf.height,
                4,
                0,
                0,
                pvcf.width,
                pvcf.height
            );

            using hl2da.pointer p = hl2da.pointer.get(image);

            System.Runtime.InteropServices.Marshal.Copy(
                p.value,
                pv_raw_buffer,
                0,
                pv_raw_buffer.Length
            );
        }
        else
        {
            System.Runtime.InteropServices.Marshal.Copy(
                fc.Buffer,
                pv_raw_buffer,
                0,
                pv_raw_buffer.Length
            );
        }

        // 上下翻转
        FlipVertical(pv_raw_buffer, pv_flip_buffer, pvcf.width, pvcf.height);

        // 写入 texture
        tex_pv.LoadRawTextureData(pv_flip_buffer);

        tex_pv.Apply(false);

        var metadata = hl2da.user.Unpack<hl2da.pv_metadata>(fb.Buffer(2));
        float[,] pose = hl2da.user.Unpack2D<float>(fb.Buffer(3), hl2da.user.POSE_ROWS, hl2da.user.POSE_COLS);
        //Matrix4x4 pose = hl2da.user.Unpack<Matrix4x4>(fb.Buffer(3));

        float[,] k_matrix = new float[,] { { metadata.fx, 0, metadata.cx }, { 0, metadata.fy, metadata.cy }, { 0, 0, 1 } };

        // encode image to png
        //byte[] frameData = ImageConversion.EncodeToPNG(tex_pv);
        //Publish(frameData, pv_width, pv_height, k_matrix, pose);
        pose_latest = CloneFloat2D(pose);
        k_latest = CloneFloat2D(k_matrix);
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
    public void SetPVUpdateEnabled(bool enabled)
    {
        _enable_sensor_update = enabled;
    }


    //void Publish(byte[] image, ushort width, ushort height, float[,] k, float[,] pose)
    //{
    //    //_publisher.PublishMessage(image, width, height, k, pose);
    //}

    public bool FreezeCurrentFrame()
    {
        if (tex_pv == null)
        {
            Game_M.initialize.XianShi("pv_freeze_ERR_tex_pv_null");
            return false;
        }

        if (tex_pv_frozen == null ||
            tex_pv_frozen.width != tex_pv.width ||
            tex_pv_frozen.height != tex_pv.height ||
            tex_pv_frozen.format != tex_pv.format)
        {
            tex_pv_frozen = new Texture2D(tex_pv.width, tex_pv.height, tex_pv.format, false);
            Game_M.initialize.XianShi("pv_freeze_01_create_tex");
        }

        tex_pv_frozen.LoadRawTextureData(tex_pv.GetRawTextureData());
        tex_pv_frozen.Apply(false);

        pose_pv_frozen = CloneFloat2D(pose_latest);
        k_pv_frozen = CloneFloat2D(k_latest);
        width_pv_frozen = (ushort)tex_pv.width;
        height_pv_frozen = (ushort)tex_pv.height;

        Game_M.initialize.XianShi("pv_freeze_02_done");
        return true;
    }
}
