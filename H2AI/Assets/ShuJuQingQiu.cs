using BestHTTP;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.OpenXR.Input;
using System.Collections;

/// <summary>
/// 数据请求
/// </summary>
public class ShuJuQingQiu : MonoBehaviour
{
    public static ShuJuQingQiu initialize;
    // Start is called before the first frame update
    public bool hasServerPose = false;
    public Vector3 serverObjectPosition = Vector3.zero;
    public Quaternion serverObjectRotation = Quaternion.identity;

    public HoloLensPVAquirer PV_controler;
    public HoloLensDepthAquirer DP_controler;

    [SerializeField] private SelectionPanelManager selectionPanelManager;

    void Start()
    {
        initialize = this;

        // =========================
        // 新增：开始采样设备位姿（ring buffer）
        // =========================
        //StartPoseSampling();
    }
    // =========================
    // 获取pv相机数据
    // =========================
    Texture2D tex_pv_P_C;
    ushort width_pv_C;
    ushort height_pv_C;
    float[,] k_pv_C;
    float[,] pose_pv_C;
    public void PublishPVMessage(Texture2D tex_pv_P, ushort width_pv, ushort height_pv, float[,] k_pv, float[,] pose_pv)
    {
        tex_pv_P_C = tex_pv_P;
        width_pv_C = width_pv;
        height_pv_C = height_pv;
        k_pv_C = k_pv;
        pose_pv_C = pose_pv;
    }

    // =========================
    // 获取dp相机数据
    // =========================
    Texture2D image_dp_P_C;
    float[,] pose_dp_C;
    public void PublishDPMessage(Texture2D image_dp_P, float[,] pose_dp)
    {
        image_dp_P_C = image_dp_P;
        pose_dp_C = pose_dp;
    }

    // =========================
    // 新增：拍照时写入（captureTicks / photoTime）
    // =========================
    long _lastCaptureTicks = 0;
    string _lastPhotoTimeUtcIso = "";
    public void NotifyCapture(long captureTicksUtcTicks, DateTime utcTime)
    {
        _lastCaptureTicks = captureTicksUtcTicks;
        _lastPhotoTimeUtcIso = utcTime.ToUniversalTime().ToString("o");
    }

    // =========================
    // 新增：设备信息（HoloLens / IP）
    // =========================
    const string DEVICE_TYPE = "HoloLens2";
    string _cachedDeviceIp = null;

    string GetDeviceIpCached()
    {
        if (_cachedDeviceIp != null) return _cachedDeviceIp;

        string ip = "";
#if WINDOWS_UWP
        try
        {
            var hostnames = Windows.Networking.Connectivity.NetworkInformation.GetHostNames();
            foreach (var hn in hostnames)
            {
                if (hn.Type == Windows.Networking.HostNameType.Ipv4)
                {
                    ip = hn.CanonicalName;
                    break;
                }
            }
        }
        catch { ip = ""; }
#endif
        _cachedDeviceIp = ip ?? "";
        return _cachedDeviceIp;
    }


    public static JArray Float2DToJArray(float[,] array)
    {
        int rows = array.GetLength(0);
        int cols = array.GetLength(1);

        JArray outer = new JArray();

        for (int i = 0; i < rows; i++)
        {
            JArray inner = new JArray();
            for (int j = 0; j < cols; j++)
            {
                inner.Add(array[i, j]);
            }
            outer.Add(inner);
        }

        return outer;
    }

    /// <summary>
    /// 上传图片
    /// </summary>
    // 上传图片

    public void ShangChuanTuPian()
    {
        StartCoroutine(ShangChuanTuPianCoroutine());
    }
    private IEnumerator ShangChuanTuPianCoroutine()
    {
        Game_M.initialize.XianShi("shangchuan");

        // ==========================================================
        // 设备相关信息
        // ==========================================================
        Game_M.initialize.XianShi("shangchuan_Device");

        string ip = GetDeviceIpCached();
        string photoTimeUtc = DateTime.UtcNow.ToString("o");
        var dev = InputDevices.GetDeviceAtXRNode(XRNode.Head);

        Vector3 headPos = Vector3.zero;
        Quaternion headRot = Quaternion.identity;

        if (dev.isValid)
        {
            dev.TryGetFeatureValue(CommonUsages.devicePosition, out headPos);
            dev.TryGetFeatureValue(CommonUsages.deviceRotation, out headRot);
        }

        PV_controler.PublishStatus = false;
        DP_controler.PublishStatus = false;
        //request.AddHeader("Content-Type", "multipart/form-data");
        // ==========================================================
        // PV图片存储与转换
        // ==========================================================
        Game_M.initialize.XianShi("shangchuan_PV");
        byte[] tex_pv_P_C_F = ImageConversion.EncodeToPNG(tex_pv_P_C);
        ushort width_pv_C_F = width_pv_C;
        ushort height_pv_C_F = height_pv_C;
        float[,] k_pv_C_F = k_pv_C;
        float[,] pose_pv_C_F = pose_pv_C;


        // ==========================================================
        // DP图片存储与转换
        // ==========================================================
        Game_M.initialize.XianShi("shangchuan_DP");
        if (image_dp_P_C == null)
        {
            Game_M.initialize.XianShi("shangchuan_image_dp_P_C_ISNULL");
            PV_controler.PublishStatus = true;
            DP_controler.PublishStatus = true;
            yield break;
        }
        byte[] image_dp_P_C_F = ImageConversion.EncodeToPNG(image_dp_P_C);
        float[,] pose_dp_C_F = pose_dp_C;
        const string SENSOR_TYPE = "AHAT";
        // ==========================================================
        // PV框选，先弹出框选窗口，等待用户确认/取消
        // ==========================================================
        Game_M.initialize.XianShi("select_box_open");

        // 这里会：
        // 1. 打开 Canvas Selection box
        // 2. 显示 tex_pv_P_C
        // 3. 初始化两个 handle
        // 4. 等用户点 Confirm / Cancel
        // 5. 自动关闭面板
        yield return StartCoroutine(
            selectionPanelManager.RequestSelection(tex_pv_P_C, headPos, headRot)
        );

        // 用户取消
        if (!selectionPanelManager.LastConfirmed)
        {
            Game_M.initialize.XianShi("select_box_cancel");

            PV_controler.PublishStatus = true;
            DP_controler.PublishStatus = true;
            yield break;
        }

        // 用户确认后的框选结果
        Vector2 boxTL = selectionPanelManager.LastTopLeftNormalized;
        Vector2 boxBR = selectionPanelManager.LastBottomRightNormalized;

        Game_M.initialize.XianShi(
            $"select_box_ok_TL({boxTL.x:F3},{boxTL.y:F3})_BR({boxBR.x:F3},{boxBR.y:F3})"
        );

        PV_controler.PublishStatus = true;
        DP_controler.PublishStatus = true;


        // ==========================================================
        // 打包
        // ==========================================================
        string url = "http://10.40.1.122:7355/generate";
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Post, OnRequestFinished);

        Game_M.initialize.XianShi("shangchuan_Dabao");
        JObject PVCameraJ = new JObject
        {
            ["image"] = Convert.ToBase64String(tex_pv_P_C_F),
            ["width"] = width_pv_C_F,
            ["height"] = height_pv_C_F,
            ["k"] = Float2DToJArray(k_pv_C_F),
            ["pose"] = Float2DToJArray(pose_pv_C_F),
        };
        request.AddField("PVCameraJ", PVCameraJ.ToString(Formatting.None));
        JObject DepthCameraJ = new JObject
        {
            ["image"] = Convert.ToBase64String(image_dp_P_C_F),
            ["pose"] = Float2DToJArray(pose_dp_C_F),
            ["sensor"] = SENSOR_TYPE,
        };
        request.AddField("DepthCameraJ", DepthCameraJ.ToString(Formatting.None));
        JObject deviceJ = new JObject
        {
            ["type"] = "DEVICE_TYPE",
            ["ip"] = string.IsNullOrEmpty(ip) ? "" : ip,
            ["time"] = photoTimeUtc,
            ["pose"] = new JArray(headPos.x, headPos.y, headPos.z),
            ["rotation"] = new JArray(headRot.x, headRot.y, headRot.z, headRot.w),
        };

        // ==========================================================
        // 新增：框选框结果（按 Python 风格：左上原点，x右，y下，0~1）
        // ==========================================================
        JObject selectionBoxJ = new JObject
        {
            ["top_left"] = new JArray(boxTL.x, boxTL.y),
            ["bottom_right"] = new JArray(boxBR.x, boxBR.y)
        };
        request.AddField("SelectionBoxJ", selectionBoxJ.ToString(Formatting.None));

        request.AddField("deviceJ", deviceJ.ToString(Formatting.None));
        request.Send();
        Game_M.initialize.XianShi("generate");
    }

    // ========================= 下面旧代码原样保留 =========================

    public string task_id;
    private void OnRequestFinished(HTTPRequest request, HTTPResponse response)
    {
        if (response.IsSuccess)
        {
            Debug.Log("Response: " + System.Text.Encoding.UTF8.GetString(response.Data));
            JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
            task_id = jo["task_id"].ToString();
            print(task_id);

            //巡检检查
            InvokeRepeating("GetJieGuo", 1, 1);
        }
        else
        {
            Debug.LogError("Error: " + response.StatusCode + " - " + response.Message);
        }
    }

    /// <summary>
    /// 检查结果
    /// </summary>
    public void GetJieGuo()
    {
        string url = "http://10.40.1.122:7355/check/?task_id=" + task_id;
        // string url = "http://10.40.1.122:7355/check";
        print(url);
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Get, OnRequestJieGuo);
        // 添加请求头数据
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        // 发送请求
        request.Send();
        Game_M.initialize.XianShi("check");
    }

    [Header("模型下载地址")]
    public string urlModel;
    [Header("图片下载地址")]
    public string image_url;
    [Header("图片")]
    public Texture2D texture2DTuPian;

    private void OnRequestJieGuo(HTTPRequest request, HTTPResponse response)
    {
        if (response.IsSuccess)
        {
            Debug.Log("Response: " + System.Text.Encoding.UTF8.GetString(response.Data));
            try
            {
                JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
                urlModel = jo["fbx_url"].ToString();//
                image_url = jo["image_url"].ToString();
                Game_M.initialize.XianShi(urlModel);
                print(urlModel);
                print(image_url);
                //关闭检测
                CancelInvoke();
                //下载模型
                XiaZaiModel();
                // XiaZaiImage();
            }
            catch (Exception)
            {
                throw;
            }

        }
        else
        {
            Debug.LogError("Error: " + response.StatusCode + " - " + response.Message);
        }
    }

    /// <summary>
    /// 下载模型
    /// </summary>
    public void XiaZaiModel()
    {
        string url = urlModel;
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Get, OnRequestXiaZai);
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.Send();
        Game_M.initialize.XianShi("download");
    }

    private void OnRequestXiaZai(HTTPRequest request, HTTPResponse response)
    {
        if (response.IsSuccess)
        {
            byte[] receiver = response.Data;
            print(receiver.Length);
            Game_M.initialize.XianShi("download " + receiver.Length);
#if !UNITY_EDITOR
            File.WriteAllBytes(Windows.Storage.ApplicationData.Current.RoamingFolder.Path + "/model.fbx", receiver);
#endif
#if UNITY_EDITOR
            File.WriteAllBytes(Application.streamingAssetsPath + "/model.fbx", receiver);
#endif
            print("保存");
            LoadModel.initialize.YanChiJiaZai();
            Game_M.initialize.XianShi("download completes");
        }
        else
        {
            Debug.LogError("Error: " + response.StatusCode + " - " + response.Message);
        }
    }

    /// <summary>
    /// 下载图片
    /// </summary>
    public void XiaZaiImage()
    {
        string url = image_url;
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Get, OnRequestImage);
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.Send();
    }

    private void OnRequestImage(HTTPRequest request, HTTPResponse response)
    {
        if (response.IsSuccess)
        {
            texture2DTuPian = response.DataAsTexture2D;
        }
        else
        {
            Debug.LogError("Error: " + response.StatusCode + " - " + response.Message);
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Q))
        {
            ShangChuanTuPian();
        }
        if (Input.GetKeyDown(KeyCode.W))
        {
            GetJieGuo();
        }
        if (Input.GetKeyDown(KeyCode.E))
        {
            XiaZaiModel();
        }
    }
}