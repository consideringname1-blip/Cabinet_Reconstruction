using BestHTTP;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Globalization;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.OpenXR.Input;
using System.Collections;

/// <summary>
/// 鏁版嵁璇锋眰
/// </summary>
public class ShuJuQingQiu : MonoBehaviour
{
    // object_reconstruction: regular reconstruction upload with depth + selection box.
    // aruco_reference: marker localization/reference upload with PV image + camera pose only.
    const string TASK_PURPOSE_OBJECT_RECONSTRUCTION = "object_reconstruction";
    const string TASK_PURPOSE_ARUCO_REFERENCE = "aruco_reference";
    const string TASK_PURPOSE_EXISTING_MODEL_REFRESH = "existing_model_refresh";
    const float MARKER_CAPTURE_TOTAL_SECONDS = 3.0f;
    const float MARKER_CAPTURE_INTERVAL_SECONDS = 0.5f;
    const int MARKER_CAPTURE_MIN_SUCCESS = 1;
    const int ARUCO_DEBUG_MARKER_RETRY_FRAMES = 30;
    const int STARTUP_CAMERA_MARKER_RETRY_FRAMES = 90;
    const int COMPLETED_MODEL_HISTORY_LIMIT = 5;
    const float ASYNC_TASK_QUEUE_POLL_INTERVAL_SECONDS = 3.0f;

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

    [Header("History Placement Restoration")]
    [SerializeField, Min(0)] private int historyPlacementRestorationModelLimit = 5;

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

    private class PendingAsyncTask
    {
        public string taskId;
        public string purpose;
        public bool modelDownloadStarted;
    }

    private RuntimeModelInstance pendingModelInstance;
    private bool pendingModelShouldPlaceDebugMarkers = true;
    private readonly Queue<PendingModelDownload> pendingModelLoadQueue = new Queue<PendingModelDownload>();
    private PendingModelDownload activeModelLoad;
    private Coroutine modelLoadQueueRetryCoroutine;
    private Coroutine arucoDebugMarkerRetryCoroutine;
    private Coroutine asyncTaskQueuePollingCoroutine;
    private readonly List<PendingAsyncTask> asyncTaskQueue = new List<PendingAsyncTask>();
    private bool asyncTaskQueueCheckInFlight = false;
    private bool asyncTaskQueuePaused = false;
    private string asyncTaskQueueActiveTaskId = "";
    private bool historyPlacementRestorationRequestInFlight = false;
    private bool historyPlacementRestorationActive = false;
    private HTTPRequest historyPlacementRestorationRequest;
    private readonly Dictionary<HTTPRequest, float> httpRequestStartedAt = new Dictionary<HTTPRequest, float>();

    void Start()
    {
        initialize = this;
        startup_session_id = BuildStartupSessionId();
        StartCoroutine(PlaceStartupCameraMarkerWhenReady());

        // =========================
        // 鏂板锛氬紑濮嬮噰鏍疯澶囦綅濮匡紙ring buffer锛?
        // =========================
        //StartPoseSampling();
    }

    private static string BuildStartupSessionId()
    {
        string timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
        string randomSuffix = Guid.NewGuid().ToString("N").Substring(0, 8);
        return timestamp + "_" + randomSuffix;
    }

    private IEnumerator PlaceStartupCameraMarkerWhenReady()
    {
        for (int frame = 0; frame < STARTUP_CAMERA_MARKER_RETRY_FRAMES; frame++)
        {
            Camera cam = Camera.main;
            CameraPoseDebugMarker marker = CameraPoseDebugMarker.Instance;
            if (cam != null && marker != null)
            {
                marker.PlaceCameraMarker(cam.transform.position, cam.transform.rotation);
                yield break;
            }

            yield return null;
        }

        Debug.LogWarning("[ARUCO] Startup camera marker could not be placed; camera or debug marker is missing.");
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

    private void LogHttpRequestStart(string label, HTTPRequest request)
    {
        if (request == null)
        {
            return;
        }

        httpRequestStartedAt[request] = Time.realtimeSinceStartup;
        string url = request.Uri != null ? request.Uri.ToString() : "";
        Debug.Log("[HTTP][REQ] " + label + " url=" + url);
    }

    private void LogHttpRequestEnd(string label, HTTPRequest request, HTTPResponse response)
    {
        float startedAt;
        string elapsed = "unknown";
        if (request != null && httpRequestStartedAt.TryGetValue(request, out startedAt))
        {
            elapsed = ((Time.realtimeSinceStartup - startedAt) * 1000f).ToString("F1", CultureInfo.InvariantCulture);
            httpRequestStartedAt.Remove(request);
        }

        string url = request != null && request.Uri != null ? request.Uri.ToString() : "";
        string statusCode = response != null ? response.StatusCode.ToString(CultureInfo.InvariantCulture) : "no_response";
        int byteCount = response != null && response.Data != null ? response.Data.Length : 0;
        bool success = response != null && response.IsSuccess;
        Debug.Log(
            "[HTTP][RESP] " + label
            + " status=" + statusCode
            + " success=" + success.ToString()
            + " elapsed_ms=" + elapsed
            + " bytes=" + byteCount.ToString(CultureInfo.InvariantCulture)
            + " url=" + url
        );
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
        if (isObjectReconstruction && lower.Contains("unsupported depth sensor"))
        {
            return "shangchuan_ERR_depth_sensor_unsupported";
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

    /// <summary>
    /// 涓婁紶鍥剧墖
    /// </summary>
    // 涓婁紶鍥剧墖

    public void ShangChuanTuPian()
    {
        ShangChuanJinJingTuPian();
    }

    public void ShangChuanJinJingTuPian()
    {
        StartObjectReconstructionCapture(DepthSensorType.AHAT);
    }

    public void ShangChuanYuanJingTuPian()
    {
        StartObjectReconstructionCapture(DepthSensorType.LONGTHROW);
    }

    private void StartObjectReconstructionCapture(DepthSensorType depthSensorType)
    {
        if (selectionPanelManager != null && selectionPanelManager.IsBusy)
        {
            Game_M.initialize.XianShi("shangchuan_ERR_selection_busy");
            return;
        }
        StartCoroutine(ShangChuanTuPianCoroutine(depthSensorType));
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

        LogHttpRequestStart("generate:" + TASK_PURPOSE_ARUCO_REFERENCE, request);
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

        LogHttpRequestStart("generate:" + purpose, request);
        request.Send();
        Game_M.initialize.XianShi("generate");
    }
    private IEnumerator ShangChuanTuPianCoroutine(DepthSensorType depthSensorType)
    {
        string requestedDepthSensorName = DP_controler != null
            ? DP_controler.GetDepthSensorName(depthSensorType)
            : depthSensorType.ToString();
        Game_M.initialize.XianShi("shangchuan_" + requestedDepthSensorName);

        // ==========================================================
        // 璁惧鐩稿叧淇℃伅
        // ==========================================================
        Game_M.initialize.XianShi("shangchuan_Device");
        bool pvFrozen = PV_controler.FreezeCurrentFrame();
        bool depthFrozen = DP_controler != null && DP_controler.FreezeCurrentFrame(depthSensorType);
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
        bool isAhatCapture = depthSensorType == DepthSensorType.AHAT;
        if (isAhatCapture && HoloLensDepthAquirer.EnableAhatUploadGuard && !DP_controler.IsFrozenAhatDepthUsable())
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
        // PV鍥剧墖瀛樺偍涓庤浆鎹?
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
        // DP鍥剧墖瀛樺偍涓庤浆鎹?
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
        if (isAhatCapture && HoloLensDepthAquirer.EnableAhatUploadGuard && image_dp_P_C_F.Length > HoloLensDepthAquirer.AHATMaxUploadPngBytes)
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
        string sensorType = DP_controler.GetPublishedDepthSensorName();
        // ==========================================================
        // PV妗嗛€夛紝鍏堝脊鍑烘閫夌獥鍙ｏ紝绛夊緟鐢ㄦ埛纭/鍙栨秷
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

        // 杩欓噷浼氾細
        // 1. 鎵撳紑 Canvas Selection box
        // 2. 鏄剧ず tex_pv_P_C
        // 3. 鍒濆鍖栦袱涓?handle
        // 4. 绛夌敤鎴风偣 Confirm / Cancel
        // 5. 鑷姩鍏抽棴闈㈡澘
        yield return StartCoroutine(
            selectionPanelManager.RequestSelection(PV_controler.tex_pv_frozen, cam.transform)
        );

        Game_M.initialize.XianShi("select_box_03_after_startcoroutine");

        // 鐢ㄦ埛鍙栨秷
        if (!selectionPanelManager.LastConfirmed)
        {
            Game_M.initialize.XianShi("select_box_cancel");
            yield break;
        }

        // 鐢ㄦ埛纭鍚庣殑妗嗛€夌粨鏋?
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
            sensorType,
            boxTL,
            boxBR
        );
    }

    public string task_id;

    private void OnRequestFinished(HTTPRequest request, HTTPResponse response)
    {
        string requestPurpose = GetRequestPurpose(request);
        LogHttpRequestEnd("generate:" + requestPurpose, request, response);
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
                Debug.Log("[ASYNC_QUEUE] queued ArUco task: " + returnedTaskId);
            }
            else
            {
                Debug.Log("[ASYNC_QUEUE] queued model task: " + returnedTaskId);
            }
            EnqueueAsyncUpdateTask(returnedTaskId, requestPurpose);
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
        asyncTaskQueue.Clear();
        asyncTaskQueuePaused = false;
        asyncTaskQueueCheckInFlight = false;
        asyncTaskQueueActiveTaskId = "";
        if (asyncTaskQueuePollingCoroutine != null)
        {
            StopCoroutine(asyncTaskQueuePollingCoroutine);
            asyncTaskQueuePollingCoroutine = null;
        }
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

        List<string> loadedTaskIds = manager.GetLoadedTaskIds();
        ModelEventDisplay eventDisplay = ModelEventDisplay.Instance;
        if (eventDisplay != null)
        {
            eventDisplay.DeleteServerEventsForTaskIds(loadedTaskIds);
            eventDisplay.CloseAllAndClearLocalCache();
        }

        int removedCount = manager.ClearLocalRuntimeModels();
        Debug.Log("[RuntimeModelManager] Cleared local runtime models: count="
            + removedCount.ToString(CultureInfo.InvariantCulture)
            + ", path="
            + manager.RuntimeModelCachePath);
        ShowFrontMessage("runtime_model_clear_" + removedCount.ToString(CultureInfo.InvariantCulture));
    }

    private void EnqueueAsyncUpdateTask(string queuedTaskId, string purpose)
    {
        EnqueueAsyncUpdateTask(queuedTaskId, purpose, false);
    }

    private void EnqueueAsyncUpdateTask(string queuedTaskId, string purpose, bool insertAtFront)
    {
        if (string.IsNullOrEmpty(queuedTaskId))
        {
            return;
        }

        string resolvedPurpose = string.IsNullOrEmpty(purpose) ? TASK_PURPOSE_OBJECT_RECONSTRUCTION : purpose;
        for (int i = asyncTaskQueue.Count - 1; i >= 0; i--)
        {
            PendingAsyncTask pendingTask = asyncTaskQueue[i];
            if (pendingTask != null && pendingTask.taskId == queuedTaskId)
            {
                if (!insertAtFront)
                {
                    return;
                }
                asyncTaskQueue.RemoveAt(i);
                break;
            }
        }

        PendingAsyncTask task = new PendingAsyncTask
        {
            taskId = queuedTaskId,
            purpose = resolvedPurpose,
        };
        if (insertAtFront)
        {
            asyncTaskQueue.Insert(0, task);
        }
        else
        {
            asyncTaskQueue.Add(task);
        }

        ShowFrontMessage("async_queue_add_" + asyncTaskQueue.Count.ToString(CultureInfo.InvariantCulture));
        EnsureAsyncTaskQueuePolling();
    }

    private void EnsureAsyncTaskQueuePolling()
    {
        if (asyncTaskQueuePollingCoroutine == null)
        {
            asyncTaskQueuePollingCoroutine = StartCoroutine(AsyncTaskQueuePollingCoroutine());
        }
    }

    private IEnumerator AsyncTaskQueuePollingCoroutine()
    {
        while (asyncTaskQueue.Count > 0 || asyncTaskQueuePaused || asyncTaskQueueCheckInFlight)
        {
            if (!asyncTaskQueuePaused && !asyncTaskQueueCheckInFlight && asyncTaskQueue.Count > 0)
            {
                SendAsyncTaskQueueCheckRequest();
            }

            yield return new WaitForSeconds(ASYNC_TASK_QUEUE_POLL_INTERVAL_SECONDS);
        }

        asyncTaskQueuePollingCoroutine = null;
    }

    private void SendAsyncTaskQueueCheckRequest()
    {
        JArray taskIds = new JArray();
        foreach (PendingAsyncTask pendingTask in asyncTaskQueue)
        {
            if (pendingTask != null && !string.IsNullOrEmpty(pendingTask.taskId))
            {
                taskIds.Add(pendingTask.taskId);
            }
        }

        if (taskIds.Count == 0)
        {
            return;
        }

        JObject payload = new JObject
        {
            ["task_ids"] = taskIds,
        };

        string url = "http://10.40.1.122:7355/check-queue";
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Post, OnAsyncTaskQueueCheckFinished);
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.RawData = Encoding.UTF8.GetBytes(payload.ToString(Formatting.None));
        asyncTaskQueueCheckInFlight = true;
        LogHttpRequestStart("check-queue", request);
        request.Send();
    }

    private bool RemoveAsyncUpdateTask(string completedTaskId, out string queuedPurpose)
    {
        queuedPurpose = "";
        if (string.IsNullOrEmpty(completedTaskId))
        {
            return false;
        }

        for (int i = 0; i < asyncTaskQueue.Count; i++)
        {
            PendingAsyncTask pendingTask = asyncTaskQueue[i];
            if (pendingTask != null && pendingTask.taskId == completedTaskId)
            {
                queuedPurpose = pendingTask.purpose;
                asyncTaskQueue.RemoveAt(i);
                return true;
            }
        }

        return false;
    }

    private string ResolvePurposeForReadyTask(string taskId, string serverPurpose, string queuedPurpose)
    {
        if (queuedPurpose == TASK_PURPOSE_EXISTING_MODEL_REFRESH)
        {
            return queuedPurpose;
        }
        if (!string.IsNullOrEmpty(serverPurpose))
        {
            return serverPurpose;
        }
        if (!string.IsNullOrEmpty(queuedPurpose))
        {
            return queuedPurpose;
        }
        return TASK_PURPOSE_OBJECT_RECONSTRUCTION;
    }

    private void ResumeAsyncTaskQueuePolling()
    {
        asyncTaskQueuePaused = false;
        asyncTaskQueueActiveTaskId = "";
        EnsureAsyncTaskQueuePolling();
    }

    private void OnAsyncTaskQueueCheckFinished(HTTPRequest request, HTTPResponse response)
    {
        LogHttpRequestEnd("check-queue", request, response);
        asyncTaskQueueCheckInFlight = false;

        if (response == null || !response.IsSuccess)
        {
            string statusCode = response != null ? response.StatusCode.ToString(CultureInfo.InvariantCulture) : "no_response";
            string message = response != null ? response.Message : "No response from server";
            Debug.LogError("[ASYNC_QUEUE] check failed: " + statusCode + " - " + message);
            ShowFrontMessage("async_queue_ERR_request_failed");
            EnsureAsyncTaskQueuePolling();
            return;
        }

        JObject wrapper = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        bool ready = wrapper["ready"] != null && wrapper["ready"].Type == JTokenType.Boolean && wrapper["ready"].Value<bool>();
        if (!ready)
        {
            HandlePendingTaskProgress(wrapper);
            EnsureAsyncTaskQueuePolling();
            return;
        }

        string completedTaskId = wrapper["task_id"]?.ToString();
        JObject taskResponse = wrapper["task"] as JObject;
        if (taskResponse == null)
        {
            Debug.LogWarning("[ASYNC_QUEUE] ready response missing task payload.");
            ShowFrontMessage("async_queue_ERR_missing_task");
            string ignoredPurpose;
            RemoveAsyncUpdateTask(completedTaskId, out ignoredPurpose);
            EnsureAsyncTaskQueuePolling();
            return;
        }

        PendingAsyncTask pendingEntry = FindPendingAsyncTask(completedTaskId);
        string queuedPurpose = pendingEntry != null ? pendingEntry.purpose : "";
        string serverPurpose = wrapper["purpose"]?.ToString() ?? taskResponse["purpose"]?.ToString();
        string purpose = ResolvePurposeForReadyTask(completedTaskId, serverPurpose, queuedPurpose);
        string status = taskResponse["status"]?.ToString() ?? wrapper["status"]?.ToString();

        if (status == "model_ready")
        {
            if (pendingEntry != null && pendingEntry.modelDownloadStarted)
            {
                EnsureAsyncTaskQueuePolling();
                return;
            }
            asyncTaskQueuePaused = true;
            asyncTaskQueueActiveTaskId = completedTaskId;
            task_id = completedTaskId;
            pendingModelShouldPlaceDebugMarkers = false;
            if (!ApplyCompletedTaskResponse(taskResponse, "ASYNC_QUEUE_MODEL_READY", false, false))
            {
                Debug.LogWarning("[ASYNC_QUEUE] model_ready response missing runtime model outputs.");
                ResumeAsyncTaskQueuePolling();
                return;
            }
            if (pendingEntry != null)
            {
                pendingEntry.modelDownloadStarted = true;
            }
            DownloadPendingRuntimeModel();
            return;
        }

        RemoveAsyncUpdateTask(completedTaskId, out queuedPurpose);
        purpose = ResolvePurposeForReadyTask(completedTaskId, serverPurpose, queuedPurpose);

        asyncTaskQueuePaused = true;
        asyncTaskQueueActiveTaskId = completedTaskId;
        task_id = completedTaskId;

        if (status == "failed")
        {
            string err = taskResponse["error"]?.ToString();
            Debug.LogError("[ASYNC_QUEUE] task failed: " + err);
            ShowFrontMessage(NormalizeServerErrorForFrontMessage(err, "check_ERR_task_failed", purpose));
            ResumeAsyncTaskQueuePolling();
            return;
        }

        if (purpose == TASK_PURPOSE_ARUCO_REFERENCE || status == "aruco_completed")
        {
            ApplyDebugInfo(taskResponse);
            bool appliedArucoReference = ApplyArucoReference(taskResponse, true, true, true);
            ShowFrontMessage(appliedArucoReference ? "aruco_completed" : "aruco_ERR_missing_reference");
            if (appliedArucoReference)
            {
                RequestModelResultAfterArucoIfNeeded(taskResponse);
            }
            ResumeAsyncTaskQueuePolling();
            return;
        }

        if (status != "completed")
        {
            Debug.LogWarning("[ASYNC_QUEUE] unsupported ready status: " + status);
            ShowFrontMessage("check_ERR_unknown_terminal_status");
            ResumeAsyncTaskQueuePolling();
            return;
        }

        if (purpose == TASK_PURPOSE_EXISTING_MODEL_REFRESH)
        {
            bool refreshedExistingModel = TryRefreshExistingRuntimeModelPoseFromResponse(taskResponse, "ASYNC_QUEUE");
            ShowFrontMessage(refreshedExistingModel ? "model_refresh_done" : "model_refresh_skip_missing_local");
            ResumeAsyncTaskQueuePolling();
            return;
        }

        HistoryPlacementRestorationDisplay evidenceDisplay = HistoryPlacementRestorationDisplay.Instance;
        if (evidenceDisplay != null)
        {
            evidenceDisplay.RegisterEvidenceForModel(taskResponse);
        }

        pendingModelShouldPlaceDebugMarkers = false;
        if (!ApplyCompletedTaskResponse(taskResponse, "ASYNC_QUEUE", false, false))
        {
            string completedError = taskResponse["error"]?.ToString();
            if (!string.IsNullOrEmpty(completedError))
            {
                Debug.LogError("[ASYNC_QUEUE] completed response missing required outputs: " + completedError);
                ShowFrontMessage(completedError);
            }
            ResumeAsyncTaskQueuePolling();
            return;
        }

        RuntimeModelManager completedManager = RuntimeModelManager.Instance;
        if (completedManager != null && completedManager.HasModel(completedTaskId))
        {
            bool refreshedExistingModel = TryRefreshExistingRuntimeModelPoseFromResponse(taskResponse, "ASYNC_QUEUE_COMPLETED");
            Debug.Log("[ASYNC_QUEUE] completed model already local; refreshed=" + refreshedExistingModel.ToString());
            ResumeAsyncTaskQueuePolling();
            return;
        }

        DownloadPendingRuntimeModel();
    }

    private void HandlePendingTaskProgress(JObject wrapper)
    {
        JArray pendingTasks = wrapper != null ? wrapper["pending"] as JArray : null;
        if (pendingTasks == null || pendingTasks.Count == 0)
        {
            return;
        }

        foreach (JToken token in pendingTasks)
        {
            JObject pendingTask = token as JObject;
            if (pendingTask == null)
            {
                continue;
            }

            string taskId = pendingTask["task_id"]?.ToString() ?? "";
            string queuedPurpose = FindQueuedPurpose(taskId);
            string purpose = ResolvePurposeForReadyTask(taskId, pendingTask["purpose"]?.ToString(), queuedPurpose);
            if (purpose != TASK_PURPOSE_OBJECT_RECONSTRUCTION)
            {
                continue;
            }

            RuntimeModelInstance hintInstance;
            if (!TryBuildPendingSpatialHintInstance(pendingTask, taskId, out hintInstance))
            {
                continue;
            }

            // Progress spatial hints are intentionally disabled; model download starts at model_ready.
        }
    }

    private PendingAsyncTask FindPendingAsyncTask(string taskId)
    {
        if (string.IsNullOrEmpty(taskId))
        {
            return null;
        }

        foreach (PendingAsyncTask pendingTask in asyncTaskQueue)
        {
            if (pendingTask != null && pendingTask.taskId == taskId)
            {
                return pendingTask;
            }
        }
        return null;
    }

    private string FindQueuedPurpose(string taskId)
    {
        if (string.IsNullOrEmpty(taskId))
        {
            return "";
        }

        foreach (PendingAsyncTask pendingTask in asyncTaskQueue)
        {
            if (pendingTask != null && pendingTask.taskId == taskId)
            {
                return pendingTask.purpose;
            }
        }
        return "";
    }

    private string BuildPendingProgressMessage(JObject pendingTask)
    {
        string progressText = pendingTask != null ? pendingTask["progress_text"]?.ToString() : "";
        if (!string.IsNullOrEmpty(progressText))
        {
            return progressText;
        }

        string status = pendingTask != null ? pendingTask["stage_name"]?.ToString() : "";
        if (string.IsNullOrEmpty(status) && pendingTask != null)
        {
            status = pendingTask["status"]?.ToString();
        }

        JToken progressToken = pendingTask != null ? pendingTask["progress"] : null;
        if (progressToken != null && progressToken.Type != JTokenType.Null)
        {
            int percent = Mathf.RoundToInt(Mathf.Clamp01(progressToken.Value<float>()) * 100f);
            if (!string.IsNullOrEmpty(status))
            {
                return status + " " + percent.ToString(CultureInfo.InvariantCulture) + "%";
            }
            return percent.ToString(CultureInfo.InvariantCulture) + "%";
        }

        return string.IsNullOrEmpty(status) ? "processing" : status;
    }

    private bool TryBuildPendingSpatialHintInstance(JObject pendingTask, string fallbackTaskId, out RuntimeModelInstance instance)
    {
        instance = null;
        if (pendingTask == null)
        {
            return false;
        }

        JObject modelJ = pendingTask["model_instance"] as JObject;
        RuntimeSpatialBoxData spatialBox;
        if (!TryParseSpatialBoxToken(
            NonNullToken(modelJ?["sam3_spatial_box"]) ?? NonNullToken(pendingTask["sam3_spatial_box"]),
            out spatialBox
        ))
        {
            return false;
        }

        string taskId = modelJ?["task_id"]?.ToString() ?? pendingTask["task_id"]?.ToString() ?? fallbackTaskId ?? "";
        string modelKey = modelJ?["model_key"]?.ToString() ?? taskId;
        if (string.IsNullOrEmpty(modelKey))
        {
            return false;
        }

        instance = new RuntimeModelInstance
        {
            ModelKey = modelKey,
            TaskId = taskId,
            FbxUrl = "",
            IsEvidenceOverlay = pendingTask["is_evidence_overlay"] != null && pendingTask["is_evidence_overlay"].Value<bool>(),
            Pose = new RuntimeModelPoseData(),
            SpatialBox = spatialBox,
        };
        return true;
    }

    private bool IsResponseArucoSynced(JObject jo)
    {
        JToken token = jo != null ? jo["aruco_coordinate_synced"] : null;
        return token != null && token.Type == JTokenType.Boolean && token.Value<bool>();
    }

    private bool TryRefreshExistingRuntimeModelPoseFromResponse(JObject jo, string sourceTag)
    {
        if (!IsResponseArucoSynced(jo))
        {
            Debug.Log("[MODEL_REFRESH] Skip unsynced completed model response from " + sourceTag + ".");
            return false;
        }

        if (!TryBuildRuntimeModelInstance(jo, out RuntimeModelInstance instance, out string errorMessage))
        {
            Debug.LogWarning("[MODEL_REFRESH] invalid completed response from " + sourceTag + ": " + errorMessage);
            return false;
        }

        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null)
        {
            ShowFrontMessage("runtime_model_mgr_missing");
            return false;
        }

        string taskId = instance.TaskId;
        string modelKey = instance.ModelKey;
        bool hasLocalModel =
            manager.HasModel(taskId)
            || manager.HasModel(modelKey);
        if (!hasLocalModel)
        {
            Debug.Log("[MODEL_REFRESH] Skip non-local completed model: task_id="
                + (string.IsNullOrEmpty(taskId) ? "<empty>" : taskId));
            return false;
        }

        string updateKey = !string.IsNullOrEmpty(taskId) ? taskId : modelKey;
        if (!manager.UpdateModelPose(updateKey, instance.Pose))
        {
            return false;
        }

        Debug.Log("[MODEL_REFRESH] Refreshed local model pose from " + sourceTag + ": " + updateKey);
        return true;
    }

    private void RequestModelResultAfterArucoIfNeeded(JObject jo)
    {
        int refreshedLoadedModelCount = EnqueueLoadedRuntimeModelRefreshesAfterAruco();
        if (refreshedLoadedModelCount <= 0)
        {
            Debug.Log("[ARUCO] Reference updated; no loaded local model needs server pose refresh.");
        }
    }

    private int EnqueueLoadedRuntimeModelRefreshesAfterAruco()
    {
        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null)
        {
            return 0;
        }

        int requestCount = 0;
        List<string> loadedTaskIds = manager.GetLoadedTaskIds();
        for (int i = loadedTaskIds.Count - 1; i >= 0; i--)
        {
            string loadedTaskId = loadedTaskIds[i];
            if (string.IsNullOrEmpty(loadedTaskId))
            {
                continue;
            }

            EnqueueAsyncUpdateTask(loadedTaskId, TASK_PURPOSE_EXISTING_MODEL_REFRESH, true);
            requestCount++;
        }

        if (requestCount > 0)
        {
            Debug.Log("[ARUCO] Refreshing loaded local models after ArUco update: count="
                + requestCount.ToString(CultureInfo.InvariantCulture));
        }
        return requestCount;
    }

    public void RefreshArucoReferenceFromServer()
    {
        string session = startup_session_id ?? "";
        if (string.IsNullOrEmpty(session))
        {
            ShowFrontMessage("aruco_ERR_no_startup_session");
            return;
        }

        string url =
            "http://10.40.1.122:7355/aruco/latest-reference?startup_session_id="
            + Uri.EscapeDataString(session);
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Get, OnRequestLatestArucoReference);
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        LogHttpRequestStart("latest-aruco-reference", request);
        request.Send();
        ShowFrontMessage("aruco_reference_refresh");
    }

    public void XiaZaiZuiXinChengGongMoXing()
    {
        QueueLatestCompletedModelsForDownload();
    }

    public void XiaZaiLiShiWuGeKeYongMoXing()
    {
        QueueLatestCompletedModelsForDownload();
    }

    public void StartHistoryPlacementRestoration()
    {
        if (historyPlacementRestorationActive || historyPlacementRestorationRequestInFlight)
        {
            historyPlacementRestorationActive = false;
            HistoryPlacementRestorationDisplay activeDisplay = HistoryPlacementRestorationDisplay.Instance;
            if (activeDisplay != null)
            {
                activeDisplay.Clear();
            }
            if (historyPlacementRestorationRequestInFlight && historyPlacementRestorationRequest != null)
            {
                historyPlacementRestorationRequest.Abort();
            }
            historyPlacementRestorationRequestInFlight = false;
            historyPlacementRestorationRequest = null;
            ShowFrontMessage("history_placement_restoration_off");
            return;
        }

        int modelLimit = Mathf.Max(0, historyPlacementRestorationModelLimit);
        JObject payload = new JObject
        {
            ["startup_session_id"] = string.IsNullOrEmpty(startup_session_id) ? "" : startup_session_id,
            ["model_limit"] = modelLimit,
        };

        string url = "http://10.40.1.122:7355/history-placement-restoration/start";
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Post, OnHistoryPlacementRestorationFinished);
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.RawData = Encoding.UTF8.GetBytes(payload.ToString(Formatting.None));
        LogHttpRequestStart("history-placement-restoration/start", request);
        historyPlacementRestorationActive = true;
        historyPlacementRestorationRequestInFlight = true;
        historyPlacementRestorationRequest = request;
        request.Send();
        ShowFrontMessage("history_placement_restoration_loading");
    }

    private int QueueHistoryPlacementRestorationModelsForDownload(JObject jo)
    {
        JArray results = jo != null ? jo["results"] as JArray : null;
        if (results == null || results.Count == 0)
        {
            return 0;
        }

        RuntimeModelManager manager = RuntimeModelManager.Instance;
        int queuedCount = 0;
        foreach (JToken token in results)
        {
            JObject result = token as JObject;
            if (result == null || result["success"] == null || !result["success"].Value<bool>())
            {
                continue;
            }

            JObject modelInstance = result["model_instance"] as JObject;
            if (modelInstance == null)
            {
                continue;
            }

            if (!ShouldAutoDownloadHistoryPlacementModel(result))
            {
                HistoryPlacementRestorationDisplay evidenceDisplayOnly = HistoryPlacementRestorationDisplay.Instance;
                if (evidenceDisplayOnly != null)
                {
                    evidenceDisplayOnly.RegisterEvidenceForModel(result);
                }
                continue;
            }

            string taskId = modelInstance["task_id"]?.ToString() ?? result["task_id"]?.ToString() ?? "";
            string modelKey = modelInstance["model_key"]?.ToString() ?? "";
            bool alreadyLoaded = manager != null
                && ((!string.IsNullOrEmpty(taskId) && manager.HasModel(taskId))
                    || (!string.IsNullOrEmpty(modelKey) && manager.HasModel(modelKey)));
            if (alreadyLoaded)
            {
                continue;
            }

            JObject downloadModel = new JObject
            {
                ["task_id"] = taskId,
                ["model_instance"] = modelInstance.DeepClone(),
            };
            foreach (string key in new[] { "object_world", "object_aruco", "aruco_reference", "sam3_spatial_box" })
            {
                JToken extra = NonNullToken(result[key]);
                if (extra != null)
                {
                    downloadModel[key] = extra.DeepClone();
                }
            }

            HistoryPlacementRestorationDisplay evidenceDisplay = HistoryPlacementRestorationDisplay.Instance;
            if (evidenceDisplay != null)
            {
                evidenceDisplay.RegisterEvidenceForModel(result);
            }

            if (DownloadRuntimeModelFromSpatialQueryModel(downloadModel))
            {
                queuedCount++;
            }
        }
        return queuedCount;
    }


    private bool ShouldAutoDownloadHistoryPlacementModel(JObject result)
    {
        if (result == null)
        {
            return false;
        }

        JObject history = result["history_placement_restoration"] as JObject;
        JObject display = history != null ? history["display"] as JObject : result["display"] as JObject;
        bool showModel = display != null && display["show_model"] != null && display["show_model"].Value<bool>();
        if (!showModel)
        {
            return false;
        }

        string status = result["status"]?.ToString() ?? history?["status"]?.ToString() ?? "";
        string normalizedStatus = status.ToUpperInvariant();
        if (normalizedStatus == "MOVED" || normalizedStatus == "ORIGINAL" || normalizedStatus == "STABLE")
        {
            return false;
        }

        string takenStatus = result["taken_object_detection"]?["status"]?.ToString() ?? "";
        if (normalizedStatus == "MISSING" && string.Equals(takenStatus, "NOT_TAKEN", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        return true;
    }

    private void OnHistoryPlacementRestorationFinished(HTTPRequest request, HTTPResponse response)
    {
        LogHttpRequestEnd("history-placement-restoration/start", request, response);
        historyPlacementRestorationRequestInFlight = false;
        if (request == historyPlacementRestorationRequest)
        {
            historyPlacementRestorationRequest = null;
        }
        if (!historyPlacementRestorationActive)
        {
            return;
        }
        if (response == null || !response.IsSuccess)
        {
            historyPlacementRestorationActive = false;
            string statusCode = response != null ? response.StatusCode.ToString(CultureInfo.InvariantCulture) : "no_response";
            string message = response != null ? response.Message : "No response from server";
            Debug.LogError("[HistoryPlacementRestoration] request failed: " + statusCode + " - " + message);
            ShowFrontMessage("history_placement_restoration_ERR_request_failed");
            return;
        }

        JObject jo;
        try
        {
            jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        }
        catch (Exception exc)
        {
            historyPlacementRestorationActive = false;
            Debug.LogError("[HistoryPlacementRestoration] invalid JSON response: " + exc.Message);
            ShowFrontMessage("history_placement_restoration_ERR_server");
            return;
        }

        bool success = jo["success"] == null || jo["success"].Value<bool>();
        if (!success)
        {
            historyPlacementRestorationActive = false;
            Debug.LogWarning("[HistoryPlacementRestoration] server returned success=false: " + response.DataAsText);
            ShowFrontMessage("history_placement_restoration_ERR_server");
            return;
        }

        HistoryPlacementRestorationDisplay display = HistoryPlacementRestorationDisplay.Instance;
        int displayCount = display != null ? display.ShowFromServerResponse(jo) : 0;
        int queuedDownloadCount = QueueHistoryPlacementRestorationModelsForDownload(jo);
        if (displayCount <= 0 && queuedDownloadCount <= 0)
        {
            historyPlacementRestorationActive = false;
        }
        int count = jo["count"] != null ? jo["count"].Value<int>() : displayCount;
        Debug.Log("[HistoryPlacementRestoration] refreshed " + count.ToString(CultureInfo.InvariantCulture)
            + " model(s), displayed " + displayCount.ToString(CultureInfo.InvariantCulture)
            + ", queued downloads " + queuedDownloadCount.ToString(CultureInfo.InvariantCulture)
            + ": " + response.DataAsText);
        if (displayCount > 0)
        {
            ShowFrontMessage("history_placement_restoration_ready_" + displayCount.ToString(CultureInfo.InvariantCulture));
        }
        else if (queuedDownloadCount > 0)
        {
            ShowFrontMessage("history_placement_restoration_downloading_model");
        }
        else
        {
            ShowFrontMessage("history_placement_restoration_no_result");
        }
    }

    private void QueueLatestCompletedModelsForDownload()
    {
        string url =
            "http://10.40.1.122:7355/latest-completed-task-ids?require_aruco_coordinate_synced=1"
            + "&limit="
            + COMPLETED_MODEL_HISTORY_LIMIT.ToString(CultureInfo.InvariantCulture);
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Get, OnLatestCompletedTaskIds);
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        LogHttpRequestStart("latest-completed-task-ids", request);
        request.Send();
        ShowFrontMessage("latest_completed_queue_loading");
    }

    private void OnLatestCompletedTaskIds(HTTPRequest request, HTTPResponse response)
    {
        LogHttpRequestEnd("latest-completed-task-ids", request, response);
        if (response == null || !response.IsSuccess)
        {
            string statusCode = response != null ? response.StatusCode.ToString(CultureInfo.InvariantCulture) : "no_response";
            string message = response != null ? response.Message : "No response from server";
            Debug.LogError("[LATEST_QUEUE] latest task ids failed: " + statusCode + " - " + message);
            ShowFrontMessage("latest_completed_ERR_request_failed");
            return;
        }

        JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        bool success = jo["success"] == null || jo["success"].Value<bool>();
        if (!success)
        {
            Debug.LogWarning("[LATEST_QUEUE] latest task ids returned success=false: " + response.DataAsText);
            ShowFrontMessage("latest_completed_ERR_request_failed");
            return;
        }

        JArray taskIds = jo["task_ids"] as JArray;
        if (taskIds == null || taskIds.Count == 0)
        {
            ShowFrontMessage("latest_completed_ERR_no_model");
            return;
        }

        int queuedCount = 0;
        foreach (JToken token in taskIds)
        {
            string latestTaskId = token.Type == JTokenType.Object
                ? token["task_id"]?.ToString()
                : token.ToString();
            if (string.IsNullOrEmpty(latestTaskId))
            {
                continue;
            }

            EnqueueAsyncUpdateTask(latestTaskId, TASK_PURPOSE_OBJECT_RECONSTRUCTION);
            queuedCount++;
        }

        ShowFrontMessage("latest_completed_queued_" + queuedCount.ToString(CultureInfo.InvariantCulture));
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

    bool TryParseSpatialBoxToken(JToken boxToken, out RuntimeSpatialBoxData spatialBox)
    {
        spatialBox = null;
        if (boxToken == null || boxToken.Type == JTokenType.Null)
        {
            return false;
        }

        string status = boxToken["status"]?.ToString() ?? "";
        string coordinateSpace = boxToken["coordinate_space"]?.ToString() ?? "unity_world";
        if (status != "ready" || coordinateSpace != "unity_world")
        {
            return false;
        }

        Vector3 minWorld;
        Vector3 maxWorld;
        if (!TryReadVector3(boxToken["aabb_min_world"], out minWorld)
            || !TryReadVector3(boxToken["aabb_max_world"], out maxWorld))
        {
            Vector3 centerWorld;
            Vector3 sizeWorld;
            if (!TryReadVector3(boxToken["center_world"], out centerWorld)
                || !TryReadVector3(boxToken["size_world"], out sizeWorld))
            {
                return false;
            }

            Vector3 half = new Vector3(
                Mathf.Abs(sizeWorld.x) * 0.5f,
                Mathf.Abs(sizeWorld.y) * 0.5f,
                Mathf.Abs(sizeWorld.z) * 0.5f
            );
            minWorld = centerWorld - half;
            maxWorld = centerWorld + half;
        }

        Vector3 sortedMin = new Vector3(
            Mathf.Min(minWorld.x, maxWorld.x),
            Mathf.Min(minWorld.y, maxWorld.y),
            Mathf.Min(minWorld.z, maxWorld.z)
        );
        Vector3 sortedMax = new Vector3(
            Mathf.Max(minWorld.x, maxWorld.x),
            Mathf.Max(minWorld.y, maxWorld.y),
            Mathf.Max(minWorld.z, maxWorld.z)
        );

        spatialBox = new RuntimeSpatialBoxData
        {
            IsReady = true,
            Status = status,
            CoordinateSpace = coordinateSpace,
            AabbMinWorld = sortedMin,
            AabbMaxWorld = sortedMax,
        };
        return true;
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

        RuntimeSpatialBoxData spatialBox = null;
        TryParseSpatialBoxToken(
            NonNullToken(modelJ["sam3_spatial_box"]) ?? NonNullToken(jo["sam3_spatial_box"]),
            out spatialBox
        );

        instance = new RuntimeModelInstance
        {
            ModelKey = modelKey,
            TaskId = modelJ["task_id"]?.ToString() ?? jo["task_id"]?.ToString() ?? "",
            FbxUrl = fbxUrl,
            IsEvidenceOverlay = modelJ["is_evidence_overlay"] != null && modelJ["is_evidence_overlay"].Value<bool>(),
            Pose = poseData,
            SpatialBox = spatialBox,
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
        if (modelJ["is_evidence_overlay"] != null)
        {
            modelInstance["is_evidence_overlay"] = modelJ["is_evidence_overlay"].DeepClone();
        }

        JToken objectWorld = NonNullToken(modelJ["object_world"]);
        JToken objectAruco = NonNullToken(modelJ["object_aruco"]);
        JToken arucoReference = NonNullToken(modelJ["aruco_reference"]);
        JToken spatialBox = NonNullToken(modelJ["sam3_spatial_box"]);
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
        if (spatialBox != null)
        {
            modelInstance["sam3_spatial_box"] = spatialBox.DeepClone();
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
        JToken spatialBox = NonNullToken(modelJ["sam3_spatial_box"]);
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
        if (spatialBox != null)
        {
            wrapper["sam3_spatial_box"] = spatialBox.DeepClone();
            if (modelInstance["sam3_spatial_box"] == null)
            {
                modelInstance["sam3_spatial_box"] = spatialBox.DeepClone();
            }
        }
        wrapper["model_instance"] = modelInstance.DeepClone();

        pendingModelShouldPlaceDebugMarkers = false;
        if (!ApplyCompletedTaskResponse(wrapper, "SPATIAL", false, false))
        {
            return false;
        }

        task_id = wrapper["task_id"]?.ToString();
        DownloadPendingRuntimeModel();
        return true;
    }

    private void OnRequestLatestArucoReference(HTTPRequest request, HTTPResponse response)
    {
        LogHttpRequestEnd("latest-aruco-reference", request, response);
        if (response == null || !response.IsSuccess)
        {
            string statusCode = response != null ? response.StatusCode.ToString(CultureInfo.InvariantCulture) : "no_response";
            string message = response != null ? response.Message : "No response from server";
            Debug.LogWarning("[ARUCO] latest-reference failed: " + statusCode + " - " + message);
            ShowFrontMessage("aruco_reference_ERR_request_failed");
            return;
        }

        JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        bool appliedArucoReference = ApplyArucoReference(jo, true, true, true);
        ShowFrontMessage(appliedArucoReference ? "aruco_reference_refreshed" : "aruco_ERR_missing_reference");
        if (appliedArucoReference)
        {
            RequestModelResultAfterArucoIfNeeded(jo);
        }
    }

    /// <summary>
    /// 涓嬭浇妯″瀷
    /// </summary>
    private void DownloadPendingRuntimeModel()
    {
        if (pendingModelInstance == null || string.IsNullOrEmpty(pendingModelInstance.FbxUrl))
        {
            ShowFrontMessage("download_ERR_missing_model_instance");
            if (asyncTaskQueuePaused)
            {
                ResumeAsyncTaskQueuePolling();
            }
            return;
        }

        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null)
        {
            Debug.LogError("[RuntimeModelManager] Missing RuntimeModelManager component on scene Scripts object.");
            ShowFrontMessage("runtime_model_mgr_missing");
            if (asyncTaskQueuePaused)
            {
                ResumeAsyncTaskQueuePolling();
            }
            return;
        }

        manager.PrepareForIncomingModel(pendingModelInstance);

        PendingModelDownload pendingDownload = new PendingModelDownload
        {
            instance = pendingModelInstance,
            localPath = manager.CreateUniqueModelPath(pendingModelInstance.ModelKey),
            showDebugMarkers = pendingModelShouldPlaceDebugMarkers,
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

        var request = new HTTPRequest(new Uri(pendingModelInstance.FbxUrl), HTTPMethods.Get, OnRequestXiaZai);
        request.Tag = pendingDownload;
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        LogHttpRequestStart("download-runtime-model", request);
        request.Send();
        ShowFrontMessage("download");
    }

    private void OnRequestXiaZai(HTTPRequest request, HTTPResponse response)
    {
        LogHttpRequestEnd("download-runtime-model", request, response);
        if (response != null && response.IsSuccess)
        {
            PendingModelDownload pendingDownload = request.Tag as PendingModelDownload;
            if (pendingDownload == null || pendingDownload.instance == null || string.IsNullOrEmpty(pendingDownload.localPath))
            {
                Debug.LogError("[DOWNLOAD] Missing pending model download metadata.");
                ShowFrontMessage("download_ERR_missing_model_instance");
                if (asyncTaskQueuePaused)
                {
                    ResumeAsyncTaskQueuePolling();
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
                if (asyncTaskQueuePaused)
                {
                    ResumeAsyncTaskQueuePolling();
                }
                return;
            }
            print(receiver.Length);
            ShowFrontMessage("download " + receiver.Length);
            File.WriteAllBytes(pendingDownload.localPath, receiver);
            print("淇濆瓨");
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
            if (asyncTaskQueuePaused)
            {
                ResumeAsyncTaskQueuePolling();
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

        if (asyncTaskQueuePaused)
        {
            ResumeAsyncTaskQueuePolling();
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

        if (asyncTaskQueuePaused)
        {
            ResumeAsyncTaskQueuePolling();
        }
    }

}
