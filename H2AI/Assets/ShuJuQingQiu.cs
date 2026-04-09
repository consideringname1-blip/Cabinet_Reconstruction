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
    public bool hasServerCameraPose = false;
    public Vector3 serverCameraPosition = Vector3.zero;
    public Quaternion serverCameraRotation = Quaternion.identity;

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
        if (selectionPanelManager != null && selectionPanelManager.IsBusy)
        {
            Game_M.initialize.XianShi("shangchuan_ERR_selection_busy");
            return;
        }
        StartCoroutine(ShangChuanTuPianCoroutine());
    }
    private IEnumerator ShangChuanTuPianCoroutine()
    {
        Game_M.initialize.XianShi("shangchuan");

        // ==========================================================
        // 设备相关信息
        // ==========================================================
        Game_M.initialize.XianShi("shangchuan_Device");
        PV_controler.FreezeCurrentFrame();
        DP_controler.FreezeCurrentFrame();


        string ip = GetDeviceIpCached();
        string photoTimeUtc = DateTime.UtcNow.ToString("o");

        Camera cam = Camera.main;
        if (cam == null)
        {
            Game_M.initialize.XianShi("select_box_no_main_camera");
            yield break;
        }
        Vector3 camPos = cam.transform.position;
        Quaternion camRot = cam.transform.rotation;
        //request.AddHeader("Content-Type", "multipart/form-data");
        // ==========================================================
        // PV图片存储与转换
        // ==========================================================
        Game_M.initialize.XianShi("shangchuan_PV");
        byte[] tex_pv_P_C_F = ImageConversion.EncodeToPNG(PV_controler.tex_pv_frozen);
        ushort width_pv_C_F = PV_controler.width_pv_frozen;
        ushort height_pv_C_F = PV_controler.height_pv_frozen;
        float[,] k_pv_C_F = PV_controler.k_pv_frozen;
        float[,] pose_pv_C_F = PV_controler.pose_pv_frozen;


        // ==========================================================
        // DP图片存储与转换
        // ==========================================================
        Game_M.initialize.XianShi("shangchuan_DP");
        if (DP_controler.tex_grayscale_publish == null)
        {
            Game_M.initialize.XianShi("shangchuan_image_dp_P_C_ISNULL");
            yield break;
        }
        byte[] image_dp_P_C_F = ImageConversion.EncodeToPNG(DP_controler.tex_grayscale_publish);
        float[,] pose_dp_C_F = DP_controler.pose_publish;
        const string SENSOR_TYPE = "AHAT";
        // ==========================================================
        // PV框选，先弹出框选窗口，等待用户确认/取消
        // ==========================================================
        Game_M.initialize.XianShi("select_box_open_before_call");

        if (selectionPanelManager == null)
        {
            Game_M.initialize.XianShi("select_box_ERR_mgr_null");
            yield break;
        }

        if (PV_controler.tex_pv_frozen == null)
        {
            Game_M.initialize.XianShi("select_box_ERR_tex_null");
            yield break;
        }

        if (cam == null)
        {
            Game_M.initialize.XianShi("select_box_ERR_cam_null_2");
            yield break;
        }

        Game_M.initialize.XianShi("select_box_02_before_startcoroutine");

        // 这里会：
        // 1. 打开 Canvas Selection box
        // 2. 显示 tex_pv_P_C
        // 3. 初始化两个 handle
        // 4. 等用户点 Confirm / Cancel
        // 5. 自动关闭面板
        yield return StartCoroutine(
            selectionPanelManager.RequestSelection(PV_controler.tex_pv_frozen, cam.transform)
        );

        Game_M.initialize.XianShi("select_box_03_after_startcoroutine");

        // 用户取消
        if (!selectionPanelManager.LastConfirmed)
        {
            Game_M.initialize.XianShi("select_box_cancel");
            yield break;
        }

        // 用户确认后的框选结果
        Vector2 boxTL = selectionPanelManager.LastTopLeftNormalized;
        Vector2 boxBR = selectionPanelManager.LastBottomRightNormalized;

        Game_M.initialize.XianShi(
            $"select_box_ok_TL({boxTL.x:F3},{boxTL.y:F3})_BR({boxBR.x:F3},{boxBR.y:F3})"
        );


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
            ["pose"] = new JArray(camPos.x, camPos.y, camPos.z),
            ["rotation"] = new JArray(camRot.x, camRot.y, camRot.z, camRot.w),
        };
        request.AddField("deviceJ", deviceJ.ToString(Formatting.None));
        // ==========================================================
        // 新增：框选框结果（按 Python 风格：左上原点，x右，y下，0~1）
        // ==========================================================
        JObject selectionBoxJ = new JObject
        {
            ["top_left"] = new JArray(boxTL.x, boxTL.y),
            ["bottom_right"] = new JArray(boxBR.x, boxBR.y)
        };
        request.AddField("SelectionBoxJ", selectionBoxJ.ToString(Formatting.None));

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

    public void XiaZaiZuiXinChengGongMoXing()
    {
        string url = "http://10.40.1.122:7355/latest-completed";
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Get, OnRequestLatestCompleted);
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.Send();
        Game_M.initialize.XianShi("latest_completed");
    }

    [Header("模型下载地址")]
    public string urlModel;
    [Header("图片下载地址")]
    public string image_url;
    [Header("图片")]
    public Texture2D texture2DTuPian;
    [Header("Debug JSON")]
    [TextArea(3, 12)]
    public string debug_json;
    [Header("Pose Transform Debug JSON")]
    [TextArea(3, 12)]
    public string pose_transform_stages_json;
    [Header("Pose Stage Debug JSON")]
    [TextArea(3, 12)]
    public string pose_stage_debug_json;
    [Header("Object Alignment Debug JSON")]
    [TextArea(3, 12)]
    public string object_alignment_debug_json;

    void ApplyDebugInfo(JObject jo)
    {
        JToken debugToken = jo["debug"];
        if (debugToken == null || debugToken.Type == JTokenType.Null)
        {
            debug_json = "";
            pose_transform_stages_json = "";
            pose_stage_debug_json = "";
            object_alignment_debug_json = "";
            return;
        }

        debug_json = debugToken.ToString(Formatting.Indented);

        JToken poseTransformStagesToken = debugToken["pose_transform_stages"];
        pose_transform_stages_json =
            poseTransformStagesToken == null || poseTransformStagesToken.Type == JTokenType.Null
            ? ""
            : poseTransformStagesToken.ToString(Formatting.Indented);

        JToken poseStageToken = poseTransformStagesToken?["pose_stage"];
        pose_stage_debug_json =
            poseStageToken == null || poseStageToken.Type == JTokenType.Null
            ? ""
            : poseStageToken.ToString(Formatting.Indented);

        JToken objectAlignmentToken = poseTransformStagesToken?["object_alignment"];
        object_alignment_debug_json =
            objectAlignmentToken == null || objectAlignmentToken.Type == JTokenType.Null
            ? ""
            : objectAlignmentToken.ToString(Formatting.Indented);
    }

    void ApplyJson(string jsonString)
    {
        JObject jo = JObject.Parse(jsonString);

        JArray pos = (JArray)jo["object"]["position"];
        JArray rot = (JArray)jo["object"]["rotation"];

        serverObjectPosition = new Vector3(
            (float)pos[0],
            (float)pos[1],
            (float)pos[2]
        );

        serverObjectRotation = new Quaternion(
            (float)rot[0],
            (float)rot[1],
            (float)rot[2],
            (float)rot[3]
        );

        JToken pvCameraPoseToken = jo["debug"]?["pose_transform_stages"]?["pose_stage"]?["pv_camera_world"]?["pose"];
        JArray pvPos = (JArray)pvCameraPoseToken?["position"];
        JArray pvRot = (JArray)pvCameraPoseToken?["rotation_quaternion_xyzw"];
        if (pvPos != null && pvPos.Count >= 3 && pvRot != null && pvRot.Count >= 4)
        {
            serverCameraPosition = new Vector3(
                (float)pvPos[0],
                (float)pvPos[1],
                (float)pvPos[2]
            );

            serverCameraRotation = new Quaternion(
                (float)pvRot[0],
                (float)pvRot[1],
                (float)pvRot[2],
                (float)pvRot[3]
            );

            hasServerCameraPose = true;
        }
        else
        {
            hasServerCameraPose = false;
        }
    }

    bool ApplyCompletedTaskResponse(JObject jo, string sourceTag)
    {
        string fbxUrl = jo["fbx_url"]?.ToString();
        string imgUrl = jo["image_url"]?.ToString();
        JToken objectToken = jo["object"];

        if (string.IsNullOrEmpty(fbxUrl))
        {
            Debug.LogWarning("[" + sourceTag + "] completed response missing fbx_url.");
            return false;
        }

        urlModel = fbxUrl;
        image_url = imgUrl;
        ApplyDebugInfo(jo);

        if (objectToken != null && objectToken.Type != JTokenType.Null)
        {
            ApplyJson(jo.ToString());
            hasServerPose = true;
        }
        else
        {
            hasServerPose = false;
            Debug.LogWarning("[" + sourceTag + "] completed response missing object.");
        }

        return true;
    }
    private void OnRequestJieGuo(HTTPRequest request, HTTPResponse response)
    {
        if (!response.IsSuccess)
        {
            Debug.LogError("Error: " + response.StatusCode + " - " + response.Message);
            return;
        }

        JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        string status = jo["status"]?.ToString();

        // 任务失败
        if (status == "failed")
        {
            string err = jo["error"]?.ToString();
            Debug.LogError("[CHECK] task failed: " + err);
            CancelInvoke();
            return;
        }

        // 还没完成，继续等下一次轮询
        if (status != "completed")
        {
            Debug.Log("[CHECK] still processing... status = " + status);
            return;
        }

        // completed 了，但结果字段还要继续检查
        string fbxUrl = jo["fbx_url"]?.ToString();
        string imgUrl = jo["image_url"]?.ToString();
        JToken objectToken = jo["object"];

        if (string.IsNullOrEmpty(fbxUrl))
        {
            Debug.LogWarning("[CHECK] completed but fbx_url is missing, keep waiting...");
            return;
        }

        urlModel = fbxUrl;
        image_url = imgUrl;
        ApplyDebugInfo(jo);

        if (objectToken != null && objectToken.Type != JTokenType.Null)
        {
            ApplyJson(jo.ToString());
            hasServerPose = true;
        }
        else
        {
            Debug.LogWarning("[CHECK] completed but object is missing.");
        }

        CancelInvoke();
        XiaZaiModel();
    }

    private void OnRequestLatestCompleted(HTTPRequest request, HTTPResponse response)
    {
        if (!response.IsSuccess)
        {
            Debug.LogError("Error: " + response.StatusCode + " - " + response.Message);
            return;
        }

        JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        string status = jo["status"]?.ToString();

        if (status != "completed")
        {
            Debug.LogWarning("[LATEST] latest-completed returned status = " + status);
            return;
        }

        if (!ApplyCompletedTaskResponse(jo, "LATEST"))
        {
            return;
        }

        task_id = jo["task_id"]?.ToString();
        XiaZaiModel();
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
            if (hasServerCameraPose && hasServerPose && CameraPoseDebugMarker.Instance != null)
            {
                CameraPoseDebugMarker.Instance.PlaceMarkers(
                    serverCameraPosition,
                    serverCameraRotation,
                    serverObjectPosition,
                    serverObjectRotation
                );
            }
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
        if (Input.GetKeyDown(KeyCode.R))
        {
            XiaZaiZuiXinChengGongMoXing();
        }
    }
}
