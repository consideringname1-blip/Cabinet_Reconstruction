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
    const int ARUCO_DEBUG_MARKER_RETRY_FRAMES = 30;
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
    [Header("Polling")]
    [SerializeField, Min(1f)] private float checkPollingIntervalSeconds = 10f;
    [SerializeField, Min(1f)] private float markerCheckPollingIntervalSeconds = 5f;
    [SerializeField, Min(1f)] private float modelQueuedPollingIntervalSeconds = 25f;
    [SerializeField, Min(1f)] private float modelProcessingPollingIntervalSeconds = 12f;
    [SerializeField, Min(1f)] private float modelCompletedPollingIntervalSeconds = 30f;
    [SerializeField, Min(0)] private int arucoLatestCompletedRetryCount = 5;
    [SerializeField, Min(0.5f)] private float arucoLatestCompletedRetryDelaySeconds = 2f;
    [SerializeField] private bool logCheckRequests = false;
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
        public bool hasDebugCameraPose;
        public Vector3 debugCameraPosition;
        public Quaternion debugCameraRotation;
        public bool hasDebugObjectPose;
        public Vector3 debugObjectPosition;
        public Quaternion debugObjectRotation;
        public bool hasDebugArucoPose;
        public Vector3 debugArucoPosition;
        public Quaternion debugArucoRotation;
    }

    private class LatestCompletedRequest
    {
        public int historyOffset;
        public bool isHistoryBatch;
        public bool isArucoRefresh;
        public int retryRemaining;
    }

    private class CheckPollRequest
    {
        public string taskId;
        public string purpose;
    }

    private RuntimeModelInstance pendingModelInstance;
    private bool pendingModelShouldPlaceDebugMarkers = true;
    private int latestCompletedHistoryOffset = 0;
    private bool isHistoryBatchDownloadActive = false;
    private int historyBatchNextOffset = 0;
    private int historyBatchLoadedCount = 0;
    private readonly Queue<PendingModelDownload> pendingModelLoadQueue = new Queue<PendingModelDownload>();
    private PendingModelDownload activeModelLoad;
    private Coroutine modelLoadQueueRetryCoroutine;
    private Coroutine arucoDebugMarkerRetryCoroutine;
    private readonly Dictionary<string, Coroutine> checkPollingCoroutinesByTaskId = new Dictionary<string, Coroutine>();
    private readonly Dictionary<string, string> checkPollingStatusByTaskId = new Dictionary<string, string>();
    private readonly Dictionary<string, int> checkPollingPositionByTaskId = new Dictionary<string, int>();
    private readonly HashSet<string> modelDownloadRequestedTaskIds = new HashSet<string>();
    private readonly HashSet<string> modelSyncedPoseAppliedTaskIds = new HashSet<string>();

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


    private static JArray Float2DToJArray(float[,] array)
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

    void PlaceArucoDebugMarkerWhenReady(Vector3 arucoPosition, Quaternion arucoRotation, string source)
    {
        if (TryPlaceArucoDebugMarker(arucoPosition, arucoRotation, source, true))
        {
            return;
        }

        if (arucoDebugMarkerRetryCoroutine != null)
        {
            StopCoroutine(arucoDebugMarkerRetryCoroutine);
        }

        arucoDebugMarkerRetryCoroutine = StartCoroutine(RetryPlaceArucoDebugMarker(arucoPosition, arucoRotation, source));
    }

    IEnumerator RetryPlaceArucoDebugMarker(Vector3 arucoPosition, Quaternion arucoRotation, string source)
    {
        for (int frame = 0; frame < ARUCO_DEBUG_MARKER_RETRY_FRAMES; frame++)
        {
            yield return null;
            if (TryPlaceArucoDebugMarker(arucoPosition, arucoRotation, source, false))
            {
                arucoDebugMarkerRetryCoroutine = null;
                yield break;
            }
        }

        arucoDebugMarkerRetryCoroutine = null;
        Debug.LogWarning(
            $"[ARUCO] showDebugMarkers=true but CameraPoseDebugMarker.Instance was not found after " +
            $"{ARUCO_DEBUG_MARKER_RETRY_FRAMES} frames ({source})."
        );
        ShowFrontMessage("aruco_ERR_debug_marker_missing");
    }

    bool TryPlaceArucoDebugMarker(
        Vector3 arucoPosition,
        Quaternion arucoRotation,
        string source,
        bool logMissing
    )
    {
        CameraPoseDebugMarker marker = CameraPoseDebugMarker.Instance;
        if (marker == null)
        {
            if (logMissing)
            {
                Debug.LogWarning(
                    $"[ARUCO] showDebugMarkers=true but CameraPoseDebugMarker.Instance is missing ({source}); retrying."
                );
            }
            return false;
        }

        marker.PlaceArucoMarker(arucoPosition, arucoRotation);
        Debug.Log(
            $"[ARUCO] showDebugMarkers placed ArUco marker ({source}) " +
            $"pos={arucoPosition}, rot={arucoRotation.eulerAngles}"
        );
        return true;
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
            yield return null;
            if (PV_controler.k_pv_frozen == null || PV_controler.pose_pv_frozen == null)
            {
                Game_M.initialize.XianShi("shangchuan_mark_ERR_pose_null");
                continue;
            }

            byte[] pvPng = ImageConversion.EncodeToPNG(PV_controler.tex_pv_frozen);
            yield return null;
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
        yield return null;
        byte[] tex_pv_P_C_F = ImageConversion.EncodeToPNG(PV_controler.tex_pv_frozen);
        yield return null;
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
        yield return null;
        byte[] image_dp_P_C_F = ImageConversion.EncodeToPNG(DP_controler.tex_grayscale_publish);
        yield return null;
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


        yield return null;
        SendGenerateRequest(
            TASK_PURPOSE_OBJECT_RECONSTRUCTION,
            tex_pv_P_C_F,
            width_pv_C_F,
            height_pv_C_F,
            k_pv_C_F,
            pose_pv_C_F,
            camPos,
            camRot,
            ip,
            photoTimeUtc,
            image_dp_P_C_F,
            pose_dp_C_F,
            SENSOR_TYPE,
            boxTL,
            boxBR
        );
    }

    public string task_id;
    private string modelTaskId = "";
    private string markerTaskId = "";

    private void OnRequestFinished(HTTPRequest request, HTTPResponse response)
    {
        string requestPurpose = GetRequestPurpose(request);
        if (response != null && response.IsSuccess)
        {
            Debug.Log("Response: " + System.Text.Encoding.UTF8.GetString(response.Data));
            JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
            string returnedTaskId = jo["task_id"]?.ToString();
            if (string.IsNullOrEmpty(returnedTaskId))
            {
                ShowFrontMessage("generate_ERR_missing_task_id");
                return;
            }

            task_id = returnedTaskId;
            print(returnedTaskId);

            if (requestPurpose == TASK_PURPOSE_ARUCO_REFERENCE)
            {
                markerTaskId = returnedTaskId;
                StartMarkerCheckPolling();
            }
            else
            {
                modelTaskId = returnedTaskId;
                StartModelCheckPolling();
            }
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

    public void ClearLocalRetainedModels()
    {
        if ((LoadModel.initialize != null && LoadModel.initialize.IsLoading) || activeModelLoad != null)
        {
            ShowFrontMessage("runtime_model_clear_busy");
            return;
        }

        pendingModelLoadQueue.Clear();
        modelDownloadRequestedTaskIds.Clear();
        modelSyncedPoseAppliedTaskIds.Clear();
        if (modelLoadQueueRetryCoroutine != null)
        {
            StopCoroutine(modelLoadQueueRetryCoroutine);
            modelLoadQueueRetryCoroutine = null;
        }

        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null)
        {
            ShowFrontMessage("runtime_model_mgr_missing");
            return;
        }

        int removedCount = manager.ClearLocalRuntimeModels();
        Debug.Log("[RuntimeModelManager] Cleared local runtime models: count="
            + removedCount.ToString(CultureInfo.InvariantCulture)
            + ", path="
            + manager.RuntimeModelCachePath);
        ShowFrontMessage("runtime_model_clear_" + removedCount.ToString(CultureInfo.InvariantCulture));
    }

    private void StartModelCheckPolling()
    {
        StartCheckPollingForTask(modelTaskId, TASK_PURPOSE_OBJECT_RECONSTRUCTION);
    }

    private void StartMarkerCheckPolling()
    {
        StartCheckPollingForTask(markerTaskId, TASK_PURPOSE_ARUCO_REFERENCE);
    }

    private void StartCheckPollingForTask(string pollTaskId, string purpose)
    {
        if (string.IsNullOrEmpty(pollTaskId))
        {
            return;
        }

        StopCheckPollingForTask(pollTaskId);
        checkPollingStatusByTaskId[pollTaskId] = "pending";
        checkPollingPositionByTaskId[pollTaskId] = -1;
        checkPollingCoroutinesByTaskId[pollTaskId] = StartCoroutine(CheckPollingCoroutine(pollTaskId, purpose));
        ShowFrontMessage((purpose == TASK_PURPOSE_ARUCO_REFERENCE ? "marker" : "model") + "_polling_start");
        SendCheckRequest(pollTaskId, purpose);
    }

    private void StopCheckPollingForTask(string pollTaskId)
    {
        if (string.IsNullOrEmpty(pollTaskId))
        {
            return;
        }

        if (checkPollingCoroutinesByTaskId.TryGetValue(pollTaskId, out Coroutine coroutine) && coroutine != null)
        {
            StopCoroutine(coroutine);
        }
        checkPollingCoroutinesByTaskId.Remove(pollTaskId);
        checkPollingStatusByTaskId.Remove(pollTaskId);
        checkPollingPositionByTaskId.Remove(pollTaskId);
    }

    private float ResolveCheckPollingInterval(string pollTaskId, string purpose)
    {
        if (purpose == TASK_PURPOSE_ARUCO_REFERENCE)
        {
            return Mathf.Max(1f, markerCheckPollingIntervalSeconds);
        }

        if (!string.IsNullOrEmpty(pollTaskId)
            && checkPollingStatusByTaskId.TryGetValue(pollTaskId, out string status))
        {
            if (status == "pending")
            {
                return Mathf.Max(1f, modelQueuedPollingIntervalSeconds);
            }

            if (status == "completed")
            {
                return Mathf.Max(1f, modelCompletedPollingIntervalSeconds);
            }

            if (!IsTerminalStatus(status))
            {
                return Mathf.Max(1f, modelProcessingPollingIntervalSeconds);
            }
        }

        return Mathf.Max(1f, checkPollingIntervalSeconds);
    }

    private bool UpdateCheckPollingState(string pollTaskId, string status, JObject jo)
    {
        if (string.IsNullOrEmpty(pollTaskId))
        {
            return false;
        }

        string normalizedStatus = string.IsNullOrEmpty(status) ? "unknown" : status;
        bool changed = !checkPollingStatusByTaskId.TryGetValue(pollTaskId, out string previousStatus)
            || previousStatus != normalizedStatus;
        checkPollingStatusByTaskId[pollTaskId] = normalizedStatus;

        int position = -1;
        JToken positionToken = jo != null ? jo["position"] : null;
        if (positionToken != null && positionToken.Type != JTokenType.Null)
        {
            position = positionToken.Value<int>();
        }
        checkPollingPositionByTaskId[pollTaskId] = position;

        return changed;
    }

    private bool IsResponseArucoSynced(JObject jo)
    {
        JToken token = jo != null ? jo["aruco_coordinate_synced"] : null;
        return token != null && token.Type == JTokenType.Boolean && token.Value<bool>();
    }

    private string GetModelBoundsStatus(JObject jo)
    {
        string status = jo?["model_bounds"]?["status"]?.ToString();
        return string.IsNullOrEmpty(status) ? "missing" : status;
    }

    private bool IsModelBoundsReadyOrFinal(string status)
    {
        return status == "ready" || status == "failed";
    }

    private bool ShouldContinueCompletedModelPolling(string pollTaskId, JObject jo)
    {
        if (string.IsNullOrEmpty(pollTaskId))
        {
            return false;
        }

        bool arucoSynced = IsResponseArucoSynced(jo);
        string boundsStatus = GetModelBoundsStatus(jo);
        if (!arucoSynced || !IsModelBoundsReadyOrFinal(boundsStatus))
        {
            return true;
        }

        return modelDownloadRequestedTaskIds.Contains(pollTaskId)
            && !modelSyncedPoseAppliedTaskIds.Contains(pollTaskId);
    }

    private bool TryApplySyncedModelPose(string pollTaskId, JObject jo)
    {
        if (string.IsNullOrEmpty(pollTaskId)
            || modelSyncedPoseAppliedTaskIds.Contains(pollTaskId)
            || !IsResponseArucoSynced(jo))
        {
            return false;
        }

        if (!TryBuildRuntimeModelInstance(jo, out RuntimeModelInstance instance, out string errorMessage))
        {
            Debug.LogWarning("[CHECK] synced model response invalid: " + errorMessage);
            return false;
        }

        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null)
        {
            ShowFrontMessage("runtime_model_mgr_missing");
            return false;
        }

        if (!manager.UpdateModelPose(pollTaskId, instance.Pose))
        {
            return false;
        }

        modelSyncedPoseAppliedTaskIds.Add(pollTaskId);
        Debug.Log("[RuntimeModelManager] Applied synced ArUco pose for task: " + pollTaskId);
        ShowFrontMessage("model_pose_synced");
        return true;
    }

    private bool HasActiveModelPolling()
    {
        return !string.IsNullOrEmpty(modelTaskId)
            && checkPollingCoroutinesByTaskId.ContainsKey(modelTaskId);
    }

    private void RequestModelResultAfterArucoIfNeeded(JObject jo)
    {
        if (HasActiveModelPolling())
        {
            SendCheckRequest(modelTaskId, TASK_PURPOSE_OBJECT_RECONSTRUCTION);
        }

        RefreshLatestCompletedModelAfterAruco();
    }

    public void RefreshLatestCompletedModelAfterAruco()
    {
        RequestLatestCompletedModel(0, false, true, Mathf.Max(0, arucoLatestCompletedRetryCount));
    }

    private IEnumerator CheckPollingCoroutine(string pollTaskId, string purpose)
    {
        while (checkPollingCoroutinesByTaskId.ContainsKey(pollTaskId))
        {
            yield return new WaitForSeconds(ResolveCheckPollingInterval(pollTaskId, purpose));
            if (!string.IsNullOrEmpty(pollTaskId))
            {
                SendCheckRequest(pollTaskId, purpose);
            }
        }
    }

    private void SendCheckRequest(string checkTaskId, string purpose)
    {
        if (string.IsNullOrEmpty(checkTaskId))
        {
            return;
        }

        string url = "http://10.40.1.122:7355/check/?task_id=" + checkTaskId;
        // string url = "http://10.40.1.122:7355/check";
        if (logCheckRequests)
        {
            Debug.Log("[CHECK] " + purpose + " " + checkTaskId);
        }
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Get, OnRequestJieGuo);
        request.Tag = new CheckPollRequest
        {
            taskId = checkTaskId,
            purpose = purpose,
        };
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.Send();
    }

    public void XiaZaiZuiXinChengGongMoXing()
    {
        RequestLatestCompletedModel(0, false);
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

    private void RequestLatestCompletedModel(int historyOffset, bool isHistoryBatch)
    {
        RequestLatestCompletedModel(historyOffset, isHistoryBatch, false, 0);
    }

    private void RequestLatestCompletedModel(
        int historyOffset,
        bool isHistoryBatch,
        bool isArucoRefresh,
        int retryRemaining
    )
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
            isArucoRefresh = isArucoRefresh,
            retryRemaining = Mathf.Max(0, retryRemaining),
        };
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.Send();
        Game_M.initialize.XianShi("latest_completed_" + (latestCompletedHistoryOffset + 1).ToString(CultureInfo.InvariantCulture));
    }

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
        JToken rotationToken = poseToken["rotation_quaternion_xyzw"];
        return TryReadVector3(positionToken, out position) && TryReadQuaternion(rotationToken, out rotation);
    }

    JToken NonNullToken(JToken token)
    {
        return token == null || token.Type == JTokenType.Null ? null : token;
    }

    JToken ResolveArucoReferenceToken(JObject jo)
    {
        JToken modelInstanceToken = NonNullToken(jo["model_instance"]);
        return NonNullToken(jo["aruco_reference"])
            ?? NonNullToken(modelInstanceToken?["aruco_reference"])
            ?? NonNullToken(jo["debug"]?["pose_transform_stages"]?["aruco_stage"]?["marker_pose_world"]);
    }

    bool ApplyArucoReference(JObject jo, bool updateCurrentSession, bool showDebugMarkers, bool warnIfMissing = false)
    {
        JToken arucoReferenceToken = ResolveArucoReferenceToken(jo);
        if (!TryParsePoseToken(arucoReferenceToken, out Vector3 arucoPosition, out Quaternion arucoRotation))
        {
            if (warnIfMissing)
            {
                Debug.LogWarning(
                    $"[ARUCO] showDebugMarkers={showDebugMarkers}, updateCurrentSession={updateCurrentSession}; " +
                    "response missing usable aruco_reference: " + jo.ToString(Formatting.None)
                );
            }
            return false;
        }

        Debug.Log(
            $"[ARUCO] Parsed aruco_reference updateCurrentSession={updateCurrentSession}, " +
            $"showDebugMarkers={showDebugMarkers}, pos={arucoPosition}, rot={arucoRotation.eulerAngles}"
        );

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

        if (showDebugMarkers)
        {
            PlaceArucoDebugMarkerWhenReady(arucoPosition, arucoRotation, "aruco_reference");
        }
        return true;
    }

    bool TryResolveObjectWorldPose(JObject jo, out Vector3 position, out Quaternion rotation)
    {
        position = Vector3.zero;
        rotation = Quaternion.identity;

        JToken modelInstanceToken = NonNullToken(jo["model_instance"]);
        JToken objectArucoToken = NonNullToken(jo["object_aruco"]) ?? NonNullToken(modelInstanceToken?["object_aruco"]);
        JToken objectWorldToken = NonNullToken(jo["object_world"]) ?? NonNullToken(modelInstanceToken?["object_world"]);
        JToken arucoReferenceToken = ResolveArucoReferenceToken(jo);
        Vector3 responseArucoPosition;
        Quaternion responseArucoRotation;
        bool hasResponseArucoReference = TryParsePoseToken(
            arucoReferenceToken,
            out responseArucoPosition,
            out responseArucoRotation
        );
        Vector3 localPosition;
        Quaternion localRotation;
        bool hasArucoObjectPose = TryParsePoseToken(objectArucoToken, out localPosition, out localRotation);
        if (hasArucoObjectPose && hasResponseArucoReference)
        {
            position = responseArucoPosition + (responseArucoRotation * localPosition);
            rotation = responseArucoRotation * localRotation;
            return true;
        }

        if (hasArucoObjectPose && hasArucoReferencePose)
        {
            position = arucoReferencePosition + (arucoReferenceRotation * localPosition);
            rotation = arucoReferenceRotation * localRotation;
            return true;
        }

        if (TryParsePoseToken(objectWorldToken, out position, out rotation))
        {
            return true;
        }

        return false;
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
        if (!TryBuildRuntimeModelInstance(jo, out RuntimeModelInstance modelInstance, out string errorMessage))
        {
            Debug.LogWarning("[" + sourceTag + "] completed response invalid model_instance: " + errorMessage);
            ShowFrontMessage(errorMessage);
            return false;
        }

        pendingModelInstance = modelInstance;
        ApplyDebugInfo(jo);
        ApplyResponsePoses(jo, updateCurrentSessionArucoReference, showDebugMarkers);

        if (!modelInstance.Pose.HasWorldPose && !modelInstance.Pose.HasArucoPose)
        {
            Debug.LogWarning("[" + sourceTag + "] completed response missing model pose.");
            ShowFrontMessage("pose_WARN_missing_object");
        }
        else if (modelInstance.Pose.HasWorldPose && !modelInstance.Pose.HasArucoPose)
        {
            Debug.Log(
                "[" + sourceTag + "] using temporary world/device pose for this startup session; "
                + "ArUco-local pose will be applied when a reference becomes available."
            );
            ShowFrontMessage("pose_world_temporary");
        }

        return true;
    }

    JObject BuildSpatialQueryModelInstance(JObject modelJ)
    {
        if (modelJ == null)
        {
            return null;
        }

        string taskId = modelJ["task_id"]?.ToString() ?? "";
        string fbxUrl = modelJ["fbx_url"]?.ToString();
        if (string.IsNullOrEmpty(fbxUrl))
        {
            fbxUrl = modelJ["download_urls"]?["fbx"]?.ToString();
        }
        if (string.IsNullOrEmpty(fbxUrl))
        {
            return null;
        }

        string modelKey = modelJ["model_key"]?.ToString();
        if (string.IsNullOrEmpty(modelKey))
        {
            modelKey = string.IsNullOrEmpty(taskId) ? fbxUrl : taskId;
        }

        JObject modelInstance = new JObject
        {
            ["model_key"] = modelKey,
            ["task_id"] = taskId,
            ["fbx_url"] = fbxUrl,
        };

        JToken objectWorld = NonNullToken(modelJ["object_world"]);
        JToken objectAruco = NonNullToken(modelJ["object_aruco"]);
        JToken arucoReference = NonNullToken(modelJ["aruco_reference"]);
        if (objectWorld != null)
        {
            modelInstance["object_world"] = objectWorld.DeepClone();
        }
        if (objectAruco != null)
        {
            modelInstance["object_aruco"] = objectAruco.DeepClone();
        }
        if (arucoReference != null)
        {
            modelInstance["aruco_reference"] = arucoReference.DeepClone();
        }

        return modelInstance;
    }

    public bool DownloadRuntimeModelFromSpatialQueryModel(JObject modelJ)
    {
        if (modelJ == null)
        {
            ShowFrontMessage("spatial_query_ERR_missing_model");
            return false;
        }

        JObject wrapper = new JObject
        {
            ["status"] = "completed",
            ["task_id"] = modelJ["task_id"]?.ToString() ?? "",
        };

        JToken modelInstanceToken = NonNullToken(modelJ["model_instance"]);
        JObject modelInstance = modelInstanceToken as JObject ?? BuildSpatialQueryModelInstance(modelJ);
        if (modelInstance == null)
        {
            ShowFrontMessage("download_ERR_missing_model_instance");
            return false;
        }
        wrapper["model_instance"] = modelInstance.DeepClone();

        JToken objectWorld = NonNullToken(modelJ["object_world"]);
        JToken objectAruco = NonNullToken(modelJ["object_aruco"]);
        JToken arucoReference = NonNullToken(modelJ["aruco_reference"]);
        if (objectWorld != null)
        {
            wrapper["object_world"] = objectWorld.DeepClone();
        }
        if (objectAruco != null)
        {
            wrapper["object_aruco"] = objectAruco.DeepClone();
        }
        if (arucoReference != null)
        {
            wrapper["aruco_reference"] = arucoReference.DeepClone();
        }

        pendingModelShouldPlaceDebugMarkers = false;
        if (!ApplyCompletedTaskResponse(wrapper, "SPATIAL", false, false))
        {
            return false;
        }

        task_id = wrapper["task_id"]?.ToString();
        DownloadPendingRuntimeModel(false, 0);
        return true;
    }

    private void OnRequestJieGuo(HTTPRequest request, HTTPResponse response)
    {
        CheckPollRequest pollRequest = request.Tag as CheckPollRequest;
        string pollPurpose = pollRequest != null && !string.IsNullOrEmpty(pollRequest.purpose)
            ? pollRequest.purpose
            : TASK_PURPOSE_OBJECT_RECONSTRUCTION;
        string pollTaskId = pollRequest != null ? pollRequest.taskId : task_id;

        if (string.IsNullOrEmpty(pollTaskId) || !checkPollingCoroutinesByTaskId.ContainsKey(pollTaskId))
        {
            return;
        }

        if (response == null || !response.IsSuccess)
        {
            string statusCode = response != null ? response.StatusCode.ToString() : "no_response";
            string message = response != null ? response.Message : "No response from server";
            Debug.LogError("Error: " + statusCode + " - " + message);
            ShowFrontMessage("check_ERR_request_failed");
            StopCheckPollingForTask(pollTaskId);
            return;
        }

        JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        string status = jo["status"]?.ToString();
        bool statusChanged = UpdateCheckPollingState(pollTaskId, status, jo);
        if (statusChanged && !IsTerminalStatus(status))
        {
            string positionText = checkPollingPositionByTaskId.TryGetValue(pollTaskId, out int position) && position > 0
                ? " position=" + position.ToString(CultureInfo.InvariantCulture)
                : "";
            Debug.Log("[CHECK] " + pollPurpose + " " + pollTaskId + " status=" + status + positionText);
        }
        bool isTerminal = IsTerminalResponse(jo, status);

        // 任务失败
        if (status == "failed")
        {
            string err = jo["error"]?.ToString();
            Debug.LogError("[CHECK] task failed: " + err);
            ShowFrontMessage(NormalizeServerErrorForFrontMessage(err, "check_ERR_task_failed", pollPurpose));
            StopCheckPollingForTask(pollTaskId);
            return;
        }

        // 还没完成，继续等下一次轮询
        if (status == "aruco_completed")
        {
            ApplyDebugInfo(jo);
            bool appliedArucoReference = ApplyArucoReference(jo, true, true, true);
            ShowFrontMessage(appliedArucoReference ? "aruco_completed" : "aruco_ERR_missing_reference");
            if (appliedArucoReference)
            {
                RequestModelResultAfterArucoIfNeeded(jo);
            }
            StopCheckPollingForTask(pollTaskId);
            return;
        }

        if (status != "completed")
        {
            if (isTerminal)
            {
                Debug.LogWarning("[CHECK] terminal response without supported handler. status = " + status);
                ShowFrontMessage("check_ERR_unknown_terminal_status");
                StopCheckPollingForTask(pollTaskId);
                return;
            }

            return;
        }

        if (pollPurpose == TASK_PURPOSE_ARUCO_REFERENCE)
        {
            ShowFrontMessage("check_ERR_marker_completed_unexpected");
            StopCheckPollingForTask(pollTaskId);
            return;
        }

        // Keep polling after the initial download until ArUco sync/model bounds finish.
        if (!modelDownloadRequestedTaskIds.Contains(pollTaskId))
        {
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
                    StopCheckPollingForTask(pollTaskId);
                }
                return;
            }

            modelDownloadRequestedTaskIds.Add(pollTaskId);
            DownloadPendingRuntimeModel();
        }
        else
        {
            TryApplySyncedModelPose(pollTaskId, jo);
        }

        if (ShouldContinueCompletedModelPolling(pollTaskId, jo))
        {
            return;
        }

        StopCheckPollingForTask(pollTaskId);
    }

    private void OnRequestLatestCompleted(HTTPRequest request, HTTPResponse response)
    {
        LatestCompletedRequest latestRequest = request.Tag as LatestCompletedRequest;
        bool isHistoryBatch = latestRequest != null && latestRequest.isHistoryBatch;
        int historyOffset = latestRequest != null ? latestRequest.historyOffset : latestCompletedHistoryOffset;
        bool isArucoRefresh = latestRequest != null && latestRequest.isArucoRefresh;
        int retryRemaining = latestRequest != null ? latestRequest.retryRemaining : 0;

        if (response == null || !response.IsSuccess)
        {
            string serverError = response != null ? response.Message : "No response from server";
            string responseText = response != null ? response.DataAsText : "";
            int statusCode = response != null ? response.StatusCode : 0;
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

            if (isArucoRefresh && !isHistoryBatch && statusCode == 404 && retryRemaining > 0)
            {
                ShowFrontMessage("latest_completed_retry_" + retryRemaining.ToString(CultureInfo.InvariantCulture));
                StartCoroutine(RetryLatestCompletedModel(historyOffset, retryRemaining - 1));
                return;
            }

            if (isHistoryBatch && statusCode == 404)
            {
                ShowFrontMessage("latest_completed_history_skip_" + (historyOffset + 1).ToString(CultureInfo.InvariantCulture));
                RequestNextHistoryBatchModel();
                return;
            }

            Debug.LogError("Error: " + statusCode.ToString(CultureInfo.InvariantCulture) + " - " + serverError);
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

    private IEnumerator RetryLatestCompletedModel(int historyOffset, int retryRemaining)
    {
        yield return new WaitForSeconds(Mathf.Max(0.5f, arucoLatestCompletedRetryDelaySeconds));
        RequestLatestCompletedModel(historyOffset, false, true, retryRemaining);
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
            hasDebugCameraPose = hasServerCameraPose,
            debugCameraPosition = serverCameraPosition,
            debugCameraRotation = serverCameraRotation,
            hasDebugObjectPose = hasServerPose,
            debugObjectPosition = serverObjectPosition,
            debugObjectRotation = serverObjectRotation,
            hasDebugArucoPose = hasArucoReferencePose,
            debugArucoPosition = arucoReferencePosition,
            debugArucoRotation = arucoReferenceRotation,
        };
        if (pendingModelInstance.Pose != null && pendingModelInstance.Pose.HasResponseArucoReference)
        {
            pendingDownload.hasDebugArucoPose = true;
            pendingDownload.debugArucoPosition = pendingModelInstance.Pose.ResponseArucoReferencePosition;
            pendingDownload.debugArucoRotation = pendingModelInstance.Pose.ResponseArucoReferenceRotation;
        }

        if (!string.IsNullOrEmpty(pendingModelInstance.TaskId))
        {
            modelDownloadRequestedTaskIds.Add(pendingModelInstance.TaskId);
        }

        var request = new HTTPRequest(new Uri(pendingModelInstance.FbxUrl), HTTPMethods.Get, OnRequestXiaZai);
        request.Tag = pendingDownload;
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.Send();
        Game_M.initialize.XianShi("download");
    }

    private void OnRequestXiaZai(HTTPRequest request, HTTPResponse response)
    {
        if (response != null && response.IsSuccess)
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
                if (!string.IsNullOrEmpty(pendingDownload.instance.TaskId))
                {
                    modelDownloadRequestedTaskIds.Remove(pendingDownload.instance.TaskId);
                }
                return;
            }
            print(receiver.Length);
            ShowFrontMessage("download " + receiver.Length);
            File.WriteAllBytes(pendingDownload.localPath, receiver);
            print("保存");
            if (pendingDownload.showDebugMarkers)
            {
                CameraPoseDebugMarker debugMarker = CameraPoseDebugMarker.Instance;
                if (debugMarker == null)
                {
                    Debug.LogWarning(
                        "[DOWNLOAD] showDebugMarkers=true but CameraPoseDebugMarker.Instance is missing; " +
                        "camera/model markers skipped."
                    );
                }
                else
                {
                    RuntimeModelManager manager = RuntimeModelManager.Instance;
                    Vector3 markerModelPosition = pendingDownload.debugObjectPosition;
                    Quaternion markerModelRotation = pendingDownload.debugObjectRotation;
                    bool hasMarkerModelPose = pendingDownload.hasDebugObjectPose;
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

                    if (pendingDownload.hasDebugCameraPose && hasMarkerModelPose)
                    {
                        debugMarker.PlaceMarkers(
                            pendingDownload.debugCameraPosition,
                            pendingDownload.debugCameraRotation,
                            markerModelPosition,
                            markerModelRotation
                        );
                    }
                    else if (hasMarkerModelPose)
                    {
                        debugMarker.PlaceModelMarker(markerModelPosition, markerModelRotation);
                    }
                }

                if (pendingDownload.hasDebugArucoPose)
                {
                    PlaceArucoDebugMarkerWhenReady(
                        pendingDownload.debugArucoPosition,
                        pendingDownload.debugArucoRotation,
                        "download"
                    );
                }
                else
                {
                    Debug.LogWarning(
                        "[DOWNLOAD] showDebugMarkers=true but no ArUco pose is cached for this download."
                    );
                }
            }
            EnqueueRuntimeModelLoad(pendingDownload);
            ShowFrontMessage("download completes");
        }
        else
        {
            string statusCode = response != null ? response.StatusCode.ToString() : "no_response";
            string message = response != null ? response.Message : "No response from server";
            Debug.LogError("Error: " + statusCode + " - " + message);
            ShowFrontMessage("download_ERR_request_failed");
            PendingModelDownload pendingDownload = request.Tag as PendingModelDownload;
            if (pendingDownload != null && pendingDownload.instance != null && !string.IsNullOrEmpty(pendingDownload.instance.TaskId))
            {
                modelDownloadRequestedTaskIds.Remove(pendingDownload.instance.TaskId);
            }
            if (pendingDownload != null && pendingDownload.isHistoryBatch)
            {
                RequestNextHistoryBatchModel();
            }
        }
    }

    private void EnqueueRuntimeModelLoad(PendingModelDownload pendingDownload)
    {
        if (pendingDownload == null || pendingDownload.instance == null || string.IsNullOrEmpty(pendingDownload.localPath))
        {
            return;
        }

        pendingModelLoadQueue.Enqueue(pendingDownload);
        ProcessNextQueuedRuntimeModelLoad();
    }

    private void ProcessNextQueuedRuntimeModelLoad()
    {
        if (activeModelLoad != null || pendingModelLoadQueue.Count == 0)
        {
            return;
        }

        LoadModel loader = LoadModel.Instance;
        if (loader.IsLoading)
        {
            if (modelLoadQueueRetryCoroutine == null)
            {
                modelLoadQueueRetryCoroutine = StartCoroutine(RetryQueuedRuntimeModelLoadWhenReady());
            }
            return;
        }

        activeModelLoad = pendingModelLoadQueue.Dequeue();
        loader.RuntimeModelLoadCompleted -= HandleQueuedRuntimeModelLoadCompleted;
        loader.RuntimeModelLoadCompleted += HandleQueuedRuntimeModelLoadCompleted;

        if (!loader.LoadRuntimeModel(activeModelLoad.instance, activeModelLoad.localPath))
        {
            loader.RuntimeModelLoadCompleted -= HandleQueuedRuntimeModelLoadCompleted;
            PendingModelDownload failedLoad = activeModelLoad;
            activeModelLoad = null;
            HandleQueuedRuntimeModelLoadFailed(failedLoad);
            ProcessNextQueuedRuntimeModelLoad();
        }
    }

    private IEnumerator RetryQueuedRuntimeModelLoadWhenReady()
    {
        while (LoadModel.Instance.IsLoading)
        {
            yield return null;
        }

        modelLoadQueueRetryCoroutine = null;
        ProcessNextQueuedRuntimeModelLoad();
    }

    private void HandleQueuedRuntimeModelLoadCompleted(RuntimeModelInstance instance, bool success)
    {
        LoadModel.Instance.RuntimeModelLoadCompleted -= HandleQueuedRuntimeModelLoadCompleted;
        PendingModelDownload completedLoad = activeModelLoad;
        activeModelLoad = null;

        if (!success)
        {
            HandleQueuedRuntimeModelLoadFailed(completedLoad);
        }
        else if (completedLoad != null)
        {
            if (completedLoad.isHistoryBatch)
            {
                historyBatchLoadedCount++;
                RequestNextHistoryBatchModel();
            }
        }

        ProcessNextQueuedRuntimeModelLoad();
    }

    private void HandleQueuedRuntimeModelLoadFailed(PendingModelDownload failedLoad)
    {
        if (failedLoad == null)
        {
            return;
        }

        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager != null)
        {
            manager.DeleteCachedFile(failedLoad.localPath);
        }

        if (failedLoad.instance != null && !string.IsNullOrEmpty(failedLoad.instance.TaskId))
        {
            modelDownloadRequestedTaskIds.Remove(failedLoad.instance.TaskId);
            modelSyncedPoseAppliedTaskIds.Remove(failedLoad.instance.TaskId);
        }

        if (failedLoad.isHistoryBatch)
        {
            RequestNextHistoryBatchModel();
        }
    }

}
