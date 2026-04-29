using BestHTTP;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Globalization;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.OpenXR.Input;
using System.Collections;

/// <summary>
/// 数据请求
/// </summary>
public class ShuJuQingQiu : MonoBehaviour
{
    // object_reconstruction: regular reconstruction upload with depth + selection box.
    // aruco_reference: marker localization/reference upload with PV image + camera pose only.
    const string TASK_PURPOSE_OBJECT_RECONSTRUCTION = "object_reconstruction";
    const string TASK_PURPOSE_ARUCO_REFERENCE = "aruco_reference";
    const float MARKER_CAPTURE_TOTAL_SECONDS = 3.0f;
    const float MARKER_CAPTURE_INTERVAL_SECONDS = 0.5f;
    const int MARKER_CAPTURE_MIN_SUCCESS = 1;
    const int COMPLETED_MODEL_HISTORY_LIMIT = 5;

    public static ShuJuQingQiu initialize;
    // Start is called before the first frame update
    public bool hasServerPose = false;
    public Vector3 serverObjectPosition = Vector3.zero;
    public Quaternion serverObjectRotation = Quaternion.identity;
    public bool hasServerCameraPose = false;
    public Vector3 serverCameraPosition = Vector3.zero;
    public Quaternion serverCameraRotation = Quaternion.identity;
    public string startup_session_id = "";
    public bool hasArucoReferencePose = false;
    public Vector3 arucoReferencePosition = Vector3.zero;
    public Quaternion arucoReferenceRotation = Quaternion.identity;

    public HoloLensPVAquirer PV_controler;
    public HoloLensDepthAquirer DP_controler;

    [SerializeField] private SelectionPanelManager selectionPanelManager;
    private bool isMarkerCaptureActive = false;

    private class MarkerCaptureFrame
    {
        public byte[] pvPng;
        public ushort pvWidth;
        public ushort pvHeight;
        public float[,] pvK;
        public float[,] pvPose;
        public Vector3 camPos;
        public Quaternion camRot;
        public string photoTimeUtc;
    }

    private class PendingModelDownload
    {
        public RuntimeModelInstance instance;
        public string localPath;
        public bool showDebugMarkers;
        public bool isHistoryBatch;
        public int historyOffset;
    }

    private class LatestCompletedRequest
    {
        public int historyOffset;
        public bool isHistoryBatch;
    }

    private RuntimeModelInstance pendingModelInstance;
    private bool pendingModelShouldPlaceDebugMarkers = true;
    private int latestCompletedHistoryOffset = 0;
    private bool isHistoryBatchDownloadActive = false;
    private int historyBatchNextOffset = 0;
    private int historyBatchLoadedCount = 0;

    void Start()
    {
        initialize = this;
        startup_session_id = BuildStartupSessionId();

        // =========================
        // 新增：开始采样设备位姿（ring buffer）
        // =========================
        //StartPoseSampling();
    }

    private static string BuildStartupSessionId()
    {
        string timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
        string randomSuffix = Guid.NewGuid().ToString("N").Substring(0, 8);
        return timestamp + "_" + randomSuffix;
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

    void ShowFrontMessage(string message)
    {
        if (string.IsNullOrEmpty(message))
        {
            return;
        }

        if (Game_M.initialize != null)
        {
            Game_M.initialize.XianShi(message);
        }
    }

    string NormalizeServerErrorForFrontMessage(string serverError, string fallbackMessage, string purpose)
    {
        string normalized = string.IsNullOrEmpty(serverError) ? "" : serverError.Trim();
        if (string.IsNullOrEmpty(normalized))
        {
            return fallbackMessage;
        }

        string lower = normalized.ToLowerInvariant();
        bool isObjectReconstruction = purpose == TASK_PURPOSE_OBJECT_RECONSTRUCTION;
        if (isObjectReconstruction && (lower.Contains("move the object closer") || lower.Contains("within 1.0 m")))
        {
            return "shangchuan_ERR_ahat_move_closer";
        }
        if (isObjectReconstruction && lower.Contains("only ahat"))
        {
            return "shangchuan_ERR_depth_sensor_not_ahat";
        }
        if (isObjectReconstruction && lower.Contains("too large") && (lower.Contains("ahat") || lower.Contains("depth")))
        {
            return "shangchuan_ERR_ahat_png_too_large_move_closer";
        }
        if (isObjectReconstruction && lower.Contains("too large"))
        {
            return "shangchuan_ERR_upload_too_large";
        }
        if (purpose == TASK_PURPOSE_ARUCO_REFERENCE && lower.Contains("too large"))
        {
            return "shangchuan_mark_ERR_upload_too_large";
        }
        return normalized;
    }

    string GetRequestPurpose(HTTPRequest request)
    {
        string purpose = request != null ? request.Tag as string : null;
        return string.IsNullOrEmpty(purpose) ? TASK_PURPOSE_OBJECT_RECONSTRUCTION : purpose;
    }

    bool IsTerminalStatus(string status)
    {
        return status == "completed" || status == "aruco_completed" || status == "failed";
    }

    bool IsTerminalResponse(JObject jo, string status)
    {
        JToken terminalToken = jo["terminal"];
        if (terminalToken != null && terminalToken.Type == JTokenType.Boolean)
        {
            return terminalToken.Value<bool>();
        }

        return IsTerminalStatus(status);
    }

    void StopCheckPolling()
    {
        isCheckPollingActive = false;
        CancelInvoke(nameof(GetJieGuo));
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

    public void ShangChuanDingWeiMarkTuPian()
    {
        if (selectionPanelManager != null && selectionPanelManager.IsBusy)
        {
            Game_M.initialize.XianShi("shangchuan_ERR_selection_busy");
            return;
        }

        if (isMarkerCaptureActive)
        {
            Game_M.initialize.XianShi("shangchuan_mark_busy");
            return;
        }

        StartCoroutine(ShangChuanDingWeiMarkTuPianCoroutine());
    }

    private IEnumerator ShangChuanDingWeiMarkTuPianCoroutine()
    {
        isMarkerCaptureActive = true;
        Game_M.initialize.XianShi("shangchuan_mark");

        string ip = GetDeviceIpCached();
        int targetFrameCount = Mathf.Max(
            MARKER_CAPTURE_MIN_SUCCESS,
            Mathf.FloorToInt(MARKER_CAPTURE_TOTAL_SECONDS / MARKER_CAPTURE_INTERVAL_SECONDS)
        );
        List<MarkerCaptureFrame> frames = new List<MarkerCaptureFrame>();

        for (int i = 0; i < targetFrameCount; i++)
        {
            float remaining = Mathf.Max(0f, MARKER_CAPTURE_TOTAL_SECONDS - (i * MARKER_CAPTURE_INTERVAL_SECONDS));
            Game_M.initialize.XianShi($"shangchuan_mark_wait_{remaining:F1}");
            yield return new WaitForSeconds(MARKER_CAPTURE_INTERVAL_SECONDS);

            Camera cam = Camera.main;
            if (cam == null)
            {
                Game_M.initialize.XianShi("select_box_no_main_camera");
                continue;
            }

            Game_M.initialize.XianShi($"shangchuan_mark_capture_{i + 1}_{targetFrameCount}");
            bool pvFrozen = PV_controler != null && PV_controler.FreezeCurrentFrame();
            if (!pvFrozen || PV_controler.tex_pv_frozen == null)
            {
                Game_M.initialize.XianShi("shangchuan_ERR_pv_freeze");
                continue;
            }
            if (PV_controler.k_pv_frozen == null || PV_controler.pose_pv_frozen == null)
            {
                Game_M.initialize.XianShi("shangchuan_mark_ERR_pose_null");
                continue;
            }

            byte[] pvPng = ImageConversion.EncodeToPNG(PV_controler.tex_pv_frozen);
            if (pvPng == null || pvPng.Length == 0)
            {
                Game_M.initialize.XianShi("shangchuan_mark_ERR_png_empty");
                continue;
            }

            frames.Add(new MarkerCaptureFrame
            {
                pvPng = pvPng,
                pvWidth = PV_controler.width_pv_frozen,
                pvHeight = PV_controler.height_pv_frozen,
                pvK = CloneFloat2D(PV_controler.k_pv_frozen),
                pvPose = CloneFloat2D(PV_controler.pose_pv_frozen),
                camPos = cam.transform.position,
                camRot = cam.transform.rotation,
                photoTimeUtc = DateTime.UtcNow.ToString("o"),
            });
        }

        if (frames.Count < MARKER_CAPTURE_MIN_SUCCESS)
        {
            Game_M.initialize.XianShi("shangchuan_mark_ERR_no_frames");
            isMarkerCaptureActive = false;
            yield break;
        }

        SendArucoBatchGenerateRequest(frames, ip);
        isMarkerCaptureActive = false;
    }

    void SendArucoBatchGenerateRequest(List<MarkerCaptureFrame> frames, string ip)
    {
        string url = "http://10.40.1.122:7355/generate";
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Post, OnRequestFinished);
        request.Tag = TASK_PURPOSE_ARUCO_REFERENCE;

        Game_M.initialize.XianShi("shangchuan_Dabao");
        request.AddField("purpose", TASK_PURPOSE_ARUCO_REFERENCE);

        JArray frameArray = new JArray();
        for (int i = 0; i < frames.Count; i++)
        {
            MarkerCaptureFrame frame = frames[i];
            JObject frameJ = new JObject
            {
                ["index"] = i,
                ["width"] = frame.pvWidth,
                ["height"] = frame.pvHeight,
                ["k"] = Float2DToJArray(frame.pvK),
                ["pose"] = Float2DToJArray(frame.pvPose),
                ["time"] = frame.photoTimeUtc,
                ["device_pose"] = new JArray(frame.camPos.x, frame.camPos.y, frame.camPos.z),
                ["device_rotation"] = new JArray(frame.camRot.x, frame.camRot.y, frame.camRot.z, frame.camRot.w),
            };
            frameArray.Add(frameJ);
            request.AddBinaryData("pv_image_" + i, frame.pvPng, "pv_" + i + ".png", "image/png");
            Debug.Log("[UPLOAD] ArUco PV frame " + i + " PNG bytes=" + frame.pvPng.Length);
        }
        request.AddField("PVCameraFramesJ", frameArray.ToString(Formatting.None));

        MarkerCaptureFrame deviceFrame = frames[frames.Count - 1];
        JObject deviceJ = new JObject
        {
            ["type"] = DEVICE_TYPE,
            ["ip"] = string.IsNullOrEmpty(ip) ? "" : ip,
            ["time"] = deviceFrame.photoTimeUtc,
            ["pose"] = new JArray(deviceFrame.camPos.x, deviceFrame.camPos.y, deviceFrame.camPos.z),
            ["rotation"] = new JArray(deviceFrame.camRot.x, deviceFrame.camRot.y, deviceFrame.camRot.z, deviceFrame.camRot.w),
            ["startup_session_id"] = startup_session_id,
        };
        request.AddField("deviceJ", deviceJ.ToString(Formatting.None));

        request.Send();
        Game_M.initialize.XianShi("generate");
    }

    void SendGenerateRequest(
        string purpose,
        byte[] texPvPng,
        ushort pvWidth,
        ushort pvHeight,
        float[,] pvK,
        float[,] pvPose,
        Vector3 camPos,
        Quaternion camRot,
        string ip,
        string photoTimeUtc,
        byte[] depthPng = null,
        float[,] depthPose = null,
        string sensorType = null,
        Vector2? boxTL = null,
        Vector2? boxBR = null
    )
    {
        string url = "http://10.40.1.122:7355/generate";
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Post, OnRequestFinished);
        request.Tag = purpose;

        Game_M.initialize.XianShi("shangchuan_Dabao");
        request.AddField("purpose", purpose);

        JObject PVCameraJ = new JObject
        {
            ["width"] = pvWidth,
            ["height"] = pvHeight,
            ["k"] = Float2DToJArray(pvK),
            ["pose"] = Float2DToJArray(pvPose),
        };
        request.AddField("PVCameraJ", PVCameraJ.ToString(Formatting.None));
        request.AddBinaryData("pv_image", texPvPng, "pv.png", "image/png");
        Debug.Log("[UPLOAD] PV PNG bytes=" + (texPvPng != null ? texPvPng.Length : 0));

        JObject deviceJ = new JObject
        {
            ["type"] = DEVICE_TYPE,
            ["ip"] = string.IsNullOrEmpty(ip) ? "" : ip,
            ["time"] = photoTimeUtc,
            ["pose"] = new JArray(camPos.x, camPos.y, camPos.z),
            ["rotation"] = new JArray(camRot.x, camRot.y, camRot.z, camRot.w),
            ["startup_session_id"] = startup_session_id,
        };
        request.AddField("deviceJ", deviceJ.ToString(Formatting.None));

        bool includeDepthPayload = purpose == TASK_PURPOSE_OBJECT_RECONSTRUCTION
            && depthPng != null
            && depthPose != null
            && !string.IsNullOrEmpty(sensorType);
        if (includeDepthPayload)
        {
            JObject DepthCameraJ = new JObject
            {
                ["pose"] = Float2DToJArray(depthPose),
                ["sensor"] = sensorType,
            };
            request.AddField("DepthCameraJ", DepthCameraJ.ToString(Formatting.None));
            request.AddBinaryData("depth_image", depthPng, "depth.png", "image/png");
            Debug.Log("[UPLOAD] Depth PNG bytes=" + (depthPng != null ? depthPng.Length : 0));
        }

        bool includeSelectionBox = purpose == TASK_PURPOSE_OBJECT_RECONSTRUCTION
            && boxTL.HasValue
            && boxBR.HasValue;
        if (includeSelectionBox)
        {
            JObject selectionBoxJ = new JObject
            {
                ["top_left"] = new JArray(boxTL.Value.x, boxTL.Value.y),
                ["bottom_right"] = new JArray(boxBR.Value.x, boxBR.Value.y)
            };
            request.AddField("SelectionBoxJ", selectionBoxJ.ToString(Formatting.None));
        }

        request.Send();
        Game_M.initialize.XianShi("generate");
    }
    private IEnumerator ShangChuanTuPianCoroutine()
    {
        Game_M.initialize.XianShi("shangchuan");

        // ==========================================================
        // 设备相关信息
        // ==========================================================
        Game_M.initialize.XianShi("shangchuan_Device");
        bool pvFrozen = PV_controler.FreezeCurrentFrame();
        bool depthFrozen = DP_controler.FreezeCurrentFrame();
        if (!pvFrozen)
        {
            Game_M.initialize.XianShi("shangchuan_ERR_pv_freeze");
            yield break;
        }
        if (!depthFrozen)
        {
            Game_M.initialize.XianShi("shangchuan_ERR_depth_freeze");
            yield break;
        }
        if (!DP_controler.IsUsingAhatSensor())
        {
            Game_M.initialize.XianShi("shangchuan_ERR_depth_sensor_not_ahat");
            yield break;
        }
        if (HoloLensDepthAquirer.EnableAhatUploadGuard && !DP_controler.IsFrozenAhatDepthUsable())
        {
            Debug.LogWarning(
                "[UPLOAD] AHAT depth rejected before upload. valid="
                + DP_controler.lastFrozenDepthValidPixels
                + " clipped="
                + DP_controler.lastFrozenDepthClippedPixels
            );
            Game_M.initialize.XianShi("shangchuan_ERR_ahat_move_closer");
            yield break;
        }


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
        if (image_dp_P_C_F == null || image_dp_P_C_F.Length == 0)
        {
            Game_M.initialize.XianShi("shangchuan_ERR_depth_png_empty");
            yield break;
        }
        if (HoloLensDepthAquirer.EnableAhatUploadGuard && image_dp_P_C_F.Length > HoloLensDepthAquirer.AHATMaxUploadPngBytes)
        {
            Debug.LogWarning(
                "[UPLOAD] AHAT depth PNG too large after sanitizing. bytes="
                + image_dp_P_C_F.Length
                + " valid="
                + DP_controler.lastFrozenDepthValidPixels
            );
            Game_M.initialize.XianShi("shangchuan_ERR_ahat_png_too_large_move_closer");
            yield break;
        }
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
        request.Tag = TASK_PURPOSE_OBJECT_RECONSTRUCTION;

        Game_M.initialize.XianShi("shangchuan_Dabao");
        request.AddField("purpose", TASK_PURPOSE_OBJECT_RECONSTRUCTION);
        JObject PVCameraJ = new JObject
        {
            ["width"] = width_pv_C_F,
            ["height"] = height_pv_C_F,
            ["k"] = Float2DToJArray(k_pv_C_F),
            ["pose"] = Float2DToJArray(pose_pv_C_F),
        };
        request.AddField("PVCameraJ", PVCameraJ.ToString(Formatting.None));
        request.AddBinaryData("pv_image", tex_pv_P_C_F, "pv.png", "image/png");
        Debug.Log("[UPLOAD] PV PNG bytes=" + (tex_pv_P_C_F != null ? tex_pv_P_C_F.Length : 0));
        JObject DepthCameraJ = new JObject
        {
            ["pose"] = Float2DToJArray(pose_dp_C_F),
            ["sensor"] = SENSOR_TYPE,
        };
        request.AddField("DepthCameraJ", DepthCameraJ.ToString(Formatting.None));
        request.AddBinaryData("depth_image", image_dp_P_C_F, "depth.png", "image/png");
        Debug.Log("[UPLOAD] Depth PNG bytes=" + (image_dp_P_C_F != null ? image_dp_P_C_F.Length : 0));
        JObject deviceJ = new JObject
        {
            ["type"] = DEVICE_TYPE,
            ["ip"] = string.IsNullOrEmpty(ip) ? "" : ip,
            ["time"] = photoTimeUtc,
            ["pose"] = new JArray(camPos.x, camPos.y, camPos.z),
            ["rotation"] = new JArray(camRot.x, camRot.y, camRot.z, camRot.w),
            ["startup_session_id"] = startup_session_id,
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
    private bool isCheckPollingActive = false;
    private string currentTaskPurpose = TASK_PURPOSE_OBJECT_RECONSTRUCTION;

    private void OnRequestFinished(HTTPRequest request, HTTPResponse response)
    {
        string requestPurpose = GetRequestPurpose(request);
        if (response != null && response.IsSuccess)
        {
            Debug.Log("Response: " + System.Text.Encoding.UTF8.GetString(response.Data));
            JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
            task_id = jo["task_id"].ToString();
            currentTaskPurpose = requestPurpose;
            print(task_id);

            //巡检检查
            StopCheckPolling();
            isCheckPollingActive = true;
            InvokeRepeating(nameof(GetJieGuo), 1, 1);
        }
        else
        {
            string serverError = response != null ? response.Message : "No response from server";
            string responseText = response != null ? response.DataAsText : "";
            if (!string.IsNullOrEmpty(responseText))
            {
                try
                {
                    JObject errorJo = (JObject)JsonConvert.DeserializeObject(responseText);
                    string detailed = errorJo?["error"]?.ToString();
                    if (!string.IsNullOrEmpty(detailed))
                    {
                        serverError = detailed;
                    }
                }
                catch
                {
                }
            }

            string statusCode = response != null ? response.StatusCode.ToString() : "no_response";
            Debug.LogError("Error: " + statusCode + " - " + serverError);
            ShowFrontMessage(NormalizeServerErrorForFrontMessage(serverError, "shangchuan_ERR_request_failed", requestPurpose));
        }
    }

    /// <summary>
    /// 检查结果
    /// </summary>
    public void GetJieGuo()
    {
        if (!isCheckPollingActive || string.IsNullOrEmpty(task_id))
        {
            return;
        }

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
        RequestLatestCompletedModel(0, false);
    }

    public void XiaZaiShangYiGeChengGongMoXing()
    {
        XiaZaiLiShiWuGeKeYongMoXing();
    }

    public void XiaZaiLiShiWuGeKeYongMoXing()
    {
        if (isHistoryBatchDownloadActive)
        {
            ShowFrontMessage("latest_completed_history_busy");
            return;
        }

        isHistoryBatchDownloadActive = true;
        historyBatchNextOffset = 0;
        historyBatchLoadedCount = 0;
        RequestNextHistoryBatchModel();
    }

    private void RequestNextHistoryBatchModel()
    {
        if (!isHistoryBatchDownloadActive)
        {
            return;
        }

        if (historyBatchNextOffset >= COMPLETED_MODEL_HISTORY_LIMIT)
        {
            ShowFrontMessage("latest_completed_history_done_" + historyBatchLoadedCount.ToString(CultureInfo.InvariantCulture));
            isHistoryBatchDownloadActive = false;
            return;
        }

        int historyOffset = historyBatchNextOffset;
        historyBatchNextOffset++;
        RequestLatestCompletedModel(historyOffset, true);
    }

    private void HandleHistoryBatchLoadCompleted(RuntimeModelInstance instance, bool success)
    {
        LoadModel.Instance.RuntimeModelLoadCompleted -= HandleHistoryBatchLoadCompleted;
        if (success)
        {
            historyBatchLoadedCount++;
        }
        RequestNextHistoryBatchModel();
    }

    public void XiaZaiXiaYiGeLiShiChengGongMoXing()
    {
        int historyOffset = latestCompletedHistoryOffset;
        XiaZaiLiShiChengGongMoXing(historyOffset);
        latestCompletedHistoryOffset = (historyOffset + 1) % COMPLETED_MODEL_HISTORY_LIMIT;
    }

    public void XiaZaiLiShiChengGongMoXing0()
    {
        XiaZaiLiShiChengGongMoXing(0);
    }

    public void XiaZaiLiShiChengGongMoXing1()
    {
        XiaZaiLiShiChengGongMoXing(1);
    }

    public void XiaZaiLiShiChengGongMoXing2()
    {
        XiaZaiLiShiChengGongMoXing(2);
    }

    public void XiaZaiLiShiChengGongMoXing3()
    {
        XiaZaiLiShiChengGongMoXing(3);
    }

    public void XiaZaiLiShiChengGongMoXing4()
    {
        XiaZaiLiShiChengGongMoXing(4);
    }

    public void XiaZaiLiShiChengGongMoXing(int historyOffset)
    {
        RequestLatestCompletedModel(historyOffset, false);
    }

    private void RequestLatestCompletedModel(int historyOffset, bool isHistoryBatch)
    {
        latestCompletedHistoryOffset = Mathf.Clamp(historyOffset, 0, COMPLETED_MODEL_HISTORY_LIMIT - 1);
        pendingModelShouldPlaceDebugMarkers = true;
        string url =
            "http://10.40.1.122:7355/latest-completed?startup_session_id="
            + Uri.EscapeDataString(startup_session_id ?? "")
            + "&require_aruco_coordinate_synced=1"
            + "&history_offset="
            + latestCompletedHistoryOffset.ToString(CultureInfo.InvariantCulture);
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Get, OnRequestLatestCompleted);
        request.Tag = new LatestCompletedRequest
        {
            historyOffset = latestCompletedHistoryOffset,
            isHistoryBatch = isHistoryBatch,
        };
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.Send();
        Game_M.initialize.XianShi("latest_completed_" + (latestCompletedHistoryOffset + 1).ToString(CultureInfo.InvariantCulture));
    }

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
    [Header("ArUco Stage Debug JSON")]
    [TextArea(3, 12)]
    public string aruco_stage_debug_json;

    void ApplyDebugInfo(JObject jo)
    {
        JToken debugToken = jo["debug"];
        if (debugToken == null || debugToken.Type == JTokenType.Null)
        {
            debug_json = "";
            pose_transform_stages_json = "";
            pose_stage_debug_json = "";
            object_alignment_debug_json = "";
            aruco_stage_debug_json = "";
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

        JToken arucoStageToken = poseTransformStagesToken?["aruco_stage"];
        aruco_stage_debug_json =
            arucoStageToken == null || arucoStageToken.Type == JTokenType.Null
            ? ""
            : arucoStageToken.ToString(Formatting.Indented);
    }

    bool TryReadVector3(JToken token, out Vector3 value)
    {
        value = Vector3.zero;
        JArray arr = token as JArray;
        if (arr == null || arr.Count < 3)
        {
            return false;
        }

        value = new Vector3(
            arr[0].Value<float>(),
            arr[1].Value<float>(),
            arr[2].Value<float>()
        );
        return true;
    }

    bool TryReadQuaternion(JToken token, out Quaternion value)
    {
        value = Quaternion.identity;
        JArray arr = token as JArray;
        if (arr == null || arr.Count < 4)
        {
            return false;
        }

        value = new Quaternion(
            arr[0].Value<float>(),
            arr[1].Value<float>(),
            arr[2].Value<float>(),
            arr[3].Value<float>()
        );
        return true;
    }

    bool TryParsePoseToken(JToken poseToken, out Vector3 position, out Quaternion rotation)
    {
        position = Vector3.zero;
        rotation = Quaternion.identity;
        if (poseToken == null || poseToken.Type == JTokenType.Null)
        {
            return false;
        }

        JToken positionToken = poseToken["position"];
        JToken rotationToken = poseToken["rotation_quaternion_xyzw"] ?? poseToken["rotation"];
        return TryReadVector3(positionToken, out position) && TryReadQuaternion(rotationToken, out rotation);
    }

    JToken NonNullToken(JToken token)
    {
        return token == null || token.Type == JTokenType.Null ? null : token;
    }

    void ApplyArucoReference(JObject jo, bool updateCurrentSession, bool showDebugMarkers)
    {
        if (!TryParsePoseToken(jo["aruco_reference"], out Vector3 arucoPosition, out Quaternion arucoRotation))
        {
            return;
        }

        if (updateCurrentSession)
        {
            arucoReferencePosition = arucoPosition;
            arucoReferenceRotation = arucoRotation;
            hasArucoReferencePose = true;
            RuntimeModelManager manager = RuntimeModelManager.Instance;
            if (manager != null)
            {
                manager.SetArucoReference(arucoPosition, arucoRotation);
            }
            else
            {
                Debug.LogError("[RuntimeModelManager] Missing RuntimeModelManager component on scene Scripts object.");
                ShowFrontMessage("runtime_model_mgr_missing");
            }
        }

        if (showDebugMarkers && CameraPoseDebugMarker.Instance != null)
        {
            CameraPoseDebugMarker.Instance.PlaceArucoMarker(arucoPosition, arucoRotation);
        }
    }

    bool TryResolveObjectWorldPose(JObject jo, out Vector3 position, out Quaternion rotation)
    {
        position = Vector3.zero;
        rotation = Quaternion.identity;

        JToken modelInstanceToken = NonNullToken(jo["model_instance"]);
        JToken objectToken = NonNullToken(jo["object"]) ?? NonNullToken(modelInstanceToken?["object_aruco"]);
        JToken objectWorldToken = NonNullToken(jo["object_world"]) ?? NonNullToken(modelInstanceToken?["object_world"]);
        JToken arucoReferenceToken = NonNullToken(jo["aruco_reference"]) ?? NonNullToken(modelInstanceToken?["aruco_reference"]);
        JObject objectJ = objectToken as JObject;
        string coordinateBasis = objectJ?["coordinate_basis"]?.ToString();
        Vector3 responseArucoPosition;
        Quaternion responseArucoRotation;
        bool hasResponseArucoReference = TryParsePoseToken(
            arucoReferenceToken,
            out responseArucoPosition,
            out responseArucoRotation
        );
        Vector3 localPosition;
        Quaternion localRotation;
        bool hasLocalObjectPose = TryParsePoseToken(objectToken, out localPosition, out localRotation);
        if (coordinateBasis == "aruco_local_x_right_y_up_z_forward")
        {
            if (hasResponseArucoReference && hasLocalObjectPose)
            {
                position = responseArucoPosition + (responseArucoRotation * localPosition);
                rotation = responseArucoRotation * localRotation;
                return true;
            }

            if (hasArucoReferencePose && hasLocalObjectPose)
            {
                position = arucoReferencePosition + (arucoReferenceRotation * localPosition);
                rotation = arucoReferenceRotation * localRotation;
                return true;
            }

            return TryParsePoseToken(objectWorldToken, out position, out rotation);
        }

        if (TryParsePoseToken(objectToken, out position, out rotation))
        {
            return true;
        }

        return TryParsePoseToken(objectWorldToken, out position, out rotation);
    }

    bool TryBuildRuntimeModelInstance(JObject jo, out RuntimeModelInstance instance, out string errorMessage)
    {
        instance = null;
        errorMessage = "";

        JObject modelJ = jo["model_instance"] as JObject;
        if (modelJ == null)
        {
            errorMessage = "download_ERR_missing_model_instance";
            return false;
        }

        string modelKey = modelJ["model_key"]?.ToString();
        string fbxUrl = modelJ["fbx_url"]?.ToString();
        if (string.IsNullOrEmpty(modelKey))
        {
            errorMessage = "download_ERR_missing_model_key";
            return false;
        }
        if (string.IsNullOrEmpty(fbxUrl))
        {
            errorMessage = "download_ERR_missing_fbx_url";
            return false;
        }

        RuntimeModelPoseData poseData = new RuntimeModelPoseData();
        if (TryParsePoseToken(modelJ["object_world"], out Vector3 worldPosition, out Quaternion worldRotation))
        {
            poseData.HasWorldPose = true;
            poseData.WorldPosition = worldPosition;
            poseData.WorldRotation = worldRotation;
        }

        if (TryParsePoseToken(modelJ["object_aruco"], out Vector3 arucoLocalPosition, out Quaternion arucoLocalRotation))
        {
            poseData.HasArucoPose = true;
            poseData.ArucoLocalPosition = arucoLocalPosition;
            poseData.ArucoLocalRotation = arucoLocalRotation;
        }

        if (TryParsePoseToken(modelJ["aruco_reference"], out Vector3 arucoReferencePosition, out Quaternion arucoReferenceRotation))
        {
            poseData.HasResponseArucoReference = true;
            poseData.ResponseArucoReferencePosition = arucoReferencePosition;
            poseData.ResponseArucoReferenceRotation = arucoReferenceRotation;
        }

        instance = new RuntimeModelInstance
        {
            ModelKey = modelKey,
            TaskId = modelJ["task_id"]?.ToString() ?? jo["task_id"]?.ToString() ?? "",
            FbxUrl = fbxUrl,
            Pose = poseData,
        };
        return true;
    }

    void ApplyResponsePoses(JObject jo, bool updateCurrentSessionArucoReference, bool showDebugMarkers)
    {
        ApplyArucoReference(jo, updateCurrentSessionArucoReference, showDebugMarkers);

        if (TryResolveObjectWorldPose(jo, out Vector3 objectPosition, out Quaternion objectRotation))
        {
            serverObjectPosition = objectPosition;
            serverObjectRotation = objectRotation;
            hasServerPose = true;
        }
        else
        {
            hasServerPose = false;
        }

        JToken pvCameraPoseToken = jo["debug"]?["pose_transform_stages"]?["pose_stage"]?["pv_camera_world"]?["pose"];
        if (TryParsePoseToken(pvCameraPoseToken, out Vector3 cameraPosition, out Quaternion cameraRotation))
        {
            serverCameraPosition = cameraPosition;
            serverCameraRotation = cameraRotation;
            hasServerCameraPose = true;
        }
        else
        {
            hasServerCameraPose = false;
        }
    }

    bool ApplyCompletedTaskResponse(
        JObject jo,
        string sourceTag,
        bool updateCurrentSessionArucoReference,
        bool showDebugMarkers
    )
    {
        string imgUrl = jo["image_url"]?.ToString();

        if (!TryBuildRuntimeModelInstance(jo, out RuntimeModelInstance modelInstance, out string errorMessage))
        {
            Debug.LogWarning("[" + sourceTag + "] completed response invalid model_instance: " + errorMessage);
            ShowFrontMessage(errorMessage);
            return false;
        }

        pendingModelInstance = modelInstance;
        image_url = imgUrl;
        ApplyDebugInfo(jo);
        ApplyResponsePoses(jo, updateCurrentSessionArucoReference, showDebugMarkers);

        if (!modelInstance.Pose.HasWorldPose && !modelInstance.Pose.HasArucoPose)
        {
            Debug.LogWarning("[" + sourceTag + "] completed response missing model pose.");
            ShowFrontMessage("pose_WARN_missing_object");
        }

        return true;
    }
    private void OnRequestJieGuo(HTTPRequest request, HTTPResponse response)
    {
        if (!isCheckPollingActive)
        {
            return;
        }

        if (!response.IsSuccess)
        {
            Debug.LogError("Error: " + response.StatusCode + " - " + response.Message);
            ShowFrontMessage("check_ERR_request_failed");
            StopCheckPolling();
            return;
        }

        JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        string status = jo["status"]?.ToString();
        bool isTerminal = IsTerminalResponse(jo, status);
        if (isTerminal)
        {
            StopCheckPolling();
        }

        // 任务失败
        if (status == "failed")
        {
            string err = jo["error"]?.ToString();
            Debug.LogError("[CHECK] task failed: " + err);
            ShowFrontMessage(NormalizeServerErrorForFrontMessage(err, "check_ERR_task_failed", currentTaskPurpose));
            StopCheckPolling();
            return;
        }

        // 还没完成，继续等下一次轮询
        if (status == "aruco_completed")
        {
            ApplyDebugInfo(jo);
            ApplyResponsePoses(jo, true, true);
            ShowFrontMessage("aruco_completed");
            StopCheckPolling();
            return;
        }

        if (status != "completed")
        {
            if (isTerminal)
            {
                Debug.LogWarning("[CHECK] terminal response without supported handler. status = " + status);
                ShowFrontMessage("check_ERR_unknown_terminal_status");
                StopCheckPolling();
                return;
            }

            Debug.Log("[CHECK] still processing... status = " + status);
            return;
        }

        // completed 了，但结果字段还要继续检查
        pendingModelShouldPlaceDebugMarkers = true;
        if (!ApplyCompletedTaskResponse(jo, "CHECK", false, true))
        {
            string completedError = jo["error"]?.ToString();
            if (!string.IsNullOrEmpty(completedError))
            {
                Debug.LogError("[CHECK] completed response missing required outputs: " + completedError);
                ShowFrontMessage(completedError);
            }

            if (isTerminal)
            {
                StopCheckPolling();
            }
            return;
        }

        StopCheckPolling();
        DownloadPendingRuntimeModel();
    }

    private void OnRequestLatestCompleted(HTTPRequest request, HTTPResponse response)
    {
        LatestCompletedRequest latestRequest = request.Tag as LatestCompletedRequest;
        bool isHistoryBatch = latestRequest != null && latestRequest.isHistoryBatch;
        int historyOffset = latestRequest != null ? latestRequest.historyOffset : latestCompletedHistoryOffset;

        if (!response.IsSuccess)
        {
            string serverError = response.Message;
            if (!string.IsNullOrEmpty(response.DataAsText))
            {
                try
                {
                    JObject errorJo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
                    string detailed = errorJo?["error"]?.ToString();
                    if (!string.IsNullOrEmpty(detailed))
                    {
                        serverError = detailed;
                    }
                }
                catch
                {
                }
            }

            if (isHistoryBatch && response.StatusCode == 404)
            {
                ShowFrontMessage("latest_completed_history_skip_" + (historyOffset + 1).ToString(CultureInfo.InvariantCulture));
                RequestNextHistoryBatchModel();
                return;
            }

            Debug.LogError("Error: " + response.StatusCode + " - " + serverError);
            if (!string.IsNullOrEmpty(serverError) && serverError.Contains("startup session"))
            {
                ShowFrontMessage("latest_completed_ERR_no_session_model");
            }
            else
            {
                ShowFrontMessage("latest_completed_ERR_request_failed");
            }
            return;
        }

        JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        string status = jo["status"]?.ToString();

        if (status != "completed")
        {
            Debug.LogWarning("[LATEST] latest-completed returned status = " + status);
            ShowFrontMessage("latest_completed_ERR_not_completed");
            if (isHistoryBatch)
            {
                RequestNextHistoryBatchModel();
            }
            return;
        }

        pendingModelShouldPlaceDebugMarkers = true;
        if (!ApplyCompletedTaskResponse(jo, "LATEST", true, true))
        {
            if (isHistoryBatch)
            {
                RequestNextHistoryBatchModel();
            }
            return;
        }

        task_id = jo["task_id"]?.ToString();
        DownloadPendingRuntimeModel(isHistoryBatch, historyOffset);
    }

    /// <summary>
    /// 下载模型
    /// </summary>
    private void DownloadPendingRuntimeModel()
    {
        DownloadPendingRuntimeModel(false, 0);
    }

    private void DownloadPendingRuntimeModel(bool isHistoryBatch, int historyOffset)
    {
        if (pendingModelInstance == null || string.IsNullOrEmpty(pendingModelInstance.FbxUrl))
        {
            ShowFrontMessage("download_ERR_missing_model_instance");
            if (isHistoryBatch)
            {
                RequestNextHistoryBatchModel();
            }
            return;
        }

        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null)
        {
            Debug.LogError("[RuntimeModelManager] Missing RuntimeModelManager component on scene Scripts object.");
            ShowFrontMessage("runtime_model_mgr_missing");
            if (isHistoryBatch)
            {
                RequestNextHistoryBatchModel();
            }
            return;
        }

        manager.PrepareForIncomingModel(pendingModelInstance);

        PendingModelDownload pendingDownload = new PendingModelDownload
        {
            instance = pendingModelInstance,
            localPath = manager.CreateUniqueModelPath(pendingModelInstance.ModelKey),
            showDebugMarkers = pendingModelShouldPlaceDebugMarkers,
            isHistoryBatch = isHistoryBatch,
            historyOffset = historyOffset,
        };

        var request = new HTTPRequest(new Uri(pendingModelInstance.FbxUrl), HTTPMethods.Get, OnRequestXiaZai);
        request.Tag = pendingDownload;
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.Send();
        Game_M.initialize.XianShi("download");
    }

    private void OnRequestXiaZai(HTTPRequest request, HTTPResponse response)
    {
        if (response.IsSuccess)
        {
            PendingModelDownload pendingDownload = request.Tag as PendingModelDownload;
            if (pendingDownload == null || pendingDownload.instance == null || string.IsNullOrEmpty(pendingDownload.localPath))
            {
                Debug.LogError("[DOWNLOAD] Missing pending model download metadata.");
                ShowFrontMessage("download_ERR_missing_model_instance");
                if (pendingDownload != null && pendingDownload.isHistoryBatch)
                {
                    RequestNextHistoryBatchModel();
                }
                return;
            }

            byte[] receiver = response.Data;
            if (receiver == null || receiver.Length == 0)
            {
                Debug.LogError("[DOWNLOAD] Runtime model download returned empty data.");
                ShowFrontMessage("download_ERR_empty_model");
                RuntimeModelManager manager = RuntimeModelManager.Instance;
                if (manager != null)
                {
                    manager.DeleteCachedFile(pendingDownload.localPath);
                }
                if (pendingDownload.isHistoryBatch)
                {
                    RequestNextHistoryBatchModel();
                }
                return;
            }
            print(receiver.Length);
            Game_M.initialize.XianShi("download " + receiver.Length);
            File.WriteAllBytes(pendingDownload.localPath, receiver);
            print("保存");
            if (pendingDownload.showDebugMarkers && CameraPoseDebugMarker.Instance != null)
            {
                RuntimeModelManager manager = RuntimeModelManager.Instance;
                Vector3 markerModelPosition = serverObjectPosition;
                Quaternion markerModelRotation = serverObjectRotation;
                bool hasMarkerModelPose = hasServerPose;
                if (manager != null && manager.TryResolveWorldPose(
                    pendingDownload.instance.Pose,
                    out Vector3 resolvedModelPosition,
                    out Quaternion resolvedModelRotation
                ))
                {
                    markerModelPosition = resolvedModelPosition;
                    markerModelRotation = resolvedModelRotation;
                    hasMarkerModelPose = true;
                }

                if (hasServerCameraPose && hasMarkerModelPose)
                {
                    CameraPoseDebugMarker.Instance.PlaceMarkers(
                        serverCameraPosition,
                        serverCameraRotation,
                        markerModelPosition,
                        markerModelRotation
                    );
                }
                else if (hasMarkerModelPose)
                {
                    CameraPoseDebugMarker.Instance.PlaceModelMarker(markerModelPosition, markerModelRotation);
                }

                if (hasArucoReferencePose)
                {
                    CameraPoseDebugMarker.Instance.PlaceArucoMarker(
                        arucoReferencePosition,
                        arucoReferenceRotation
                    );
                }
            }
            LoadModel loader = LoadModel.Instance;
            if (pendingDownload.isHistoryBatch)
            {
                loader.RuntimeModelLoadCompleted -= HandleHistoryBatchLoadCompleted;
                loader.RuntimeModelLoadCompleted += HandleHistoryBatchLoadCompleted;
            }

            if (!loader.LoadRuntimeModel(pendingDownload.instance, pendingDownload.localPath))
            {
                if (pendingDownload.isHistoryBatch)
                {
                    loader.RuntimeModelLoadCompleted -= HandleHistoryBatchLoadCompleted;
                }
                RuntimeModelManager manager = RuntimeModelManager.Instance;
                if (manager != null)
                {
                    manager.DeleteCachedFile(pendingDownload.localPath);
                }
                if (pendingDownload.isHistoryBatch)
                {
                    RequestNextHistoryBatchModel();
                }
                return;
            }
            Game_M.initialize.XianShi("download completes");
        }
        else
        {
            Debug.LogError("Error: " + response.StatusCode + " - " + response.Message);
            ShowFrontMessage("download_ERR_request_failed");
            PendingModelDownload pendingDownload = request.Tag as PendingModelDownload;
            if (pendingDownload != null && pendingDownload.isHistoryBatch)
            {
                RequestNextHistoryBatchModel();
            }
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
            ShowFrontMessage("image_ERR_request_failed");
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
            DownloadPendingRuntimeModel();
        }
        if (Input.GetKeyDown(KeyCode.R))
        {
            XiaZaiZuiXinChengGongMoXing();
        }
    }
}
