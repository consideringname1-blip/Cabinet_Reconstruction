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
    private hl2da.pv_captureformat pvcf;

    private Texture2D tex_pv;
    private Texture2D tex_pv_publish;
    private byte[] publish_flip_buffer;

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
        //pv_image.GetComponent<Renderer>().material.mainTexture = tex_pv;
        

        // 传出专用纹理（做上下翻转后再发）
        tex_pv_publish = new Texture2D(pvcf.width, pvcf.height, TextureFormat.BGRA32, false);
        publish_flip_buffer = new byte[pvcf.width * pvcf.height * 4];

        pv_image.texture = tex_pv_publish;
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

    private void FlipTextureVertically(Texture2D src, Texture2D dst, byte[] flipBuffer)
    {
        int width = src.width;
        int height = src.height;
        int bytesPerPixel = 4; // BGRA32
        int rowBytes = width * bytesPerPixel;

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

    void UpdateFrame()
    {
        using var fb = hl2da.framebuffer.GetFrame(hl2da.SENSOR_ID.PV, -1);
        if (fb.Status != hl2da.STATUS.OK) { return; }

        uint stride = hl2da.converter.GetStride_PV(pvcf.width);

        hl2da.converter fc = hl2da.converter.Convert(fb.Buffer(0), stride, pvcf.height, hl2da.IMT_Format.Nv12, hl2da.IMT_Format.Bgra8);

        if (stride != pvcf.width)
        {
            byte[,,] image = hl2da.coprocessor.Crop<byte>(fc.Buffer, (int)stride, pvcf.height, 4, 0, 0, pvcf.width, pvcf.height);
            using hl2da.pointer p = hl2da.pointer.get(image);
            tex_pv.LoadRawTextureData(p.value, pvcf.width * pvcf.height * 4);
        }
        else
        {
            tex_pv.LoadRawTextureData(fc.Buffer, fc.Length);
        }

        tex_pv.Apply();

        var metadata = hl2da.user.Unpack<hl2da.pv_metadata>(fb.Buffer(2));
        float[,] pose = hl2da.user.Unpack2D<float>(fb.Buffer(3), hl2da.user.POSE_ROWS, hl2da.user.POSE_COLS);
        //Matrix4x4 pose = hl2da.user.Unpack<Matrix4x4>(fb.Buffer(3));

        float[,] k_matrix = new float[,] { { metadata.fx, 0, metadata.cx }, { 0, metadata.fy, metadata.cy }, { 0, 0, 1 } };

        // encode image to png
        //byte[] frameData = ImageConversion.EncodeToPNG(tex_pv);
        //Publish(frameData, pv_width, pv_height, k_matrix, pose);
        Publish(tex_pv, pv_width, pv_height, k_matrix, pose);
    }

    //void Publish(byte[] image, ushort width, ushort height, float[,] k, float[,] pose)
    //{
    //    //_publisher.PublishMessage(image, width, height, k, pose);
    //}

    public bool PublishStatus = true;
    void Publish(Texture2D tex_pv_P, ushort width, ushort height, float[,] k, float[,] pose)
    {
        if (!PublishStatus) return;

        // 只在传出前做一次上下翻转
        FlipTextureVertically(tex_pv_P, tex_pv_publish, publish_flip_buffer);

        _publisher.PublishPVMessage(tex_pv_publish, width, height, k, pose);
    }
}
