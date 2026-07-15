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

public enum HistoryTrackingMode
{
    Live,
    EnteringHistory,
    History,
    EnteringLive,
}

/// <summary>
/// Coordinates HoloLens capture uploads, task polling, and runtime model delivery.
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
    const int STARTUP_CAMERA_MARKER_RETRY_FRAMES = 90;
    const float ASYNC_TASK_QUEUE_POLL_INTERVAL_SECONDS = 3.0f;
    const float SHIGURE_BOX_POLL_INTERVAL_SECONDS = 1.0f;
    const string DEFAULT_REALTIME_TRACKING_MODE_URL =
        "http://10.40.1.122:7355/realtime-tracking/mode";
    const string DEFAULT_REALTIME_TRACKING_STATUS_URL =
        "http://10.40.1.122:7355/realtime-tracking/status";

    public static ShuJuQingQiu initialize;
    // Start is called before the first frame update
    public bool hasServerPose = false;
    public Vector3 serverObjectPosition = Vector3.zero;
    public Quaternion serverObjectRotation = Quaternion.identity;
    public bool hasServerCameraPose = false;
    public Vector3 serverCameraPosition = Vector3.zero;
    public Quaternion serverCameraRotation = Quaternion.identity;
    public string startup_session_id = "";

    public HoloLensPVAquirer PV_controler;
    public HoloLensDepthAquirer DP_controler;

    [SerializeField] private SelectionPanelManager selectionPanelManager;

    [Header("Realtime Tracking Mode")]
    [SerializeField] private string realtimeTrackingModeUrl =
        DEFAULT_REALTIME_TRACKING_MODE_URL;
    [SerializeField] private string realtimeTrackingStatusUrl =
        DEFAULT_REALTIME_TRACKING_STATUS_URL;
    [SerializeField, Min(0.25f)] private float realtimeTrackingStatusPollIntervalSeconds = 1.0f;

    private bool isMarkerCaptureActive = false;

    private class MarkerCaptureFrame
    {
        public byte[] pvPng;
        public ushort pvWidth;
        public ushort pvHeight;
        public float[,] pvK;
        public float[,] pvPose;
        public string photoTimeUtc;
    }

    private class PendingModelDownload
    {
        public RuntimeModelInstance instance;
        public string localPath;
        public string downloadIdentityKey;
        public bool mayResumeAsyncTaskQueue;
        public bool resumeAsyncTaskQueueWhenComplete;
        public bool isRealtimeTrackingDownload;
        public bool hideWhenComplete;
        public bool loadedFromCache;
        public bool cancelledByLocalHide;
        public long displayGeneration;
        public bool showDebugMarkers;
        public bool hasDebugCameraPose;
        public Vector3 debugCameraPosition;
        public Quaternion debugCameraRotation;
        public bool hasDebugObjectPose;
        public Vector3 debugObjectPosition;
        public Quaternion debugObjectRotation;
    }

    private class PendingAsyncTask
    {
        public string taskId;
        public string purpose;
        public long displayGeneration;
    }

    private class GenerateRequestContext
    {
        public string purpose;
        public long displayGeneration;
    }

    private class HistoryTrackingModeRequestContext
    {
        public long requestGeneration;
        public HistoryTrackingMode targetMode;
        public HistoryTrackingMode previousStableMode;
        public bool isAutomaticRehandshake;
    }

    private class RealtimeTrackingStatusRequestContext
    {
        public long transportGeneration;
        public long requestGeneration;
        public long modeEpoch;
    }

    private RuntimeModelInstance pendingModelInstance;
    private long pendingModelDisplayGeneration = 0;
    private bool pendingModelShouldPlaceDebugMarkers = true;
    private readonly Queue<PendingModelDownload> pendingModelLoadQueue = new Queue<PendingModelDownload>();
    private PendingModelDownload activeModelLoad;
    private Coroutine modelLoadQueueRetryCoroutine;
    private Coroutine asyncTaskQueuePollingCoroutine;
    private Coroutine realtimeTrackingStatusPollingCoroutine;
    private Coroutine objectTrackingBoxPollingCoroutine;
    private readonly List<PendingAsyncTask> asyncTaskQueue = new List<PendingAsyncTask>();
    private readonly HashSet<string> runtimeModelDownloadsInFlight = new HashSet<string>(StringComparer.Ordinal);
    private readonly HashSet<HTTPRequest> realtimeTrackingModelDownloadRequests = new HashSet<HTTPRequest>();
    private bool asyncTaskQueueCheckInFlight = false;
    private bool asyncTaskQueuePaused = false;
    private HistoryTrackingMode historyTrackingMode = HistoryTrackingMode.Live;
    private HTTPRequest historyTrackingModeRequest;
    private HTTPRequest realtimeTrackingStatusRequest;
    private HTTPRequest objectTrackingBoxRequest;
    private bool realtimeTrackingStatusDeliveryEnabled = true;
    private bool automaticModelDeliverySuppressed = false;
    private long realtimeTrackingTransportGeneration = 0;
    private long localModelDisplayGeneration = 0;
    private long historyTrackingRequestGeneration = 0;
    private long acceptedHistoryTrackingModeEpoch = -1;
    private bool startupArUcoReferenceReady = false;
    public HistoryTrackingMode CurrentHistoryTrackingMode
    {
        get { return historyTrackingMode; }
    }

    public bool TryGetServerServiceBaseUri(out Uri serviceBaseUri)
    {
        serviceBaseUri = null;
        string[] endpointCandidates =
        {
            ResolveRealtimeTrackingStatusUrl(),
            ResolveRealtimeTrackingModeUrl(),
        };
        string[] endpointSuffixes =
        {
            "/realtime-tracking/status",
            "/realtime-tracking/mode",
        };

        foreach (string endpointCandidate in endpointCandidates)
        {
            if (!Uri.TryCreate(
                    endpointCandidate,
                    UriKind.Absolute,
                    out Uri endpointUri)
                || (endpointUri.Scheme != Uri.UriSchemeHttp
                    && endpointUri.Scheme != Uri.UriSchemeHttps))
            {
                continue;
            }

            string endpointPath = endpointUri.AbsolutePath.TrimEnd('/');
            foreach (string endpointSuffix in endpointSuffixes)
            {
                if (!endpointPath.EndsWith(
                        endpointSuffix,
                        StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                string servicePath = endpointPath.Substring(
                    0,
                    endpointPath.Length - endpointSuffix.Length);
                UriBuilder builder = new UriBuilder(endpointUri)
                {
                    Path = string.IsNullOrEmpty(servicePath)
                        ? "/"
                        : servicePath.TrimEnd('/') + "/",
                    Query = "",
                    Fragment = "",
                };
                serviceBaseUri = builder.Uri;
                return true;
            }
        }
        return false;
    }

    private string ResolveRealtimeTrackingModeUrl()
    {
        return string.IsNullOrWhiteSpace(realtimeTrackingModeUrl)
            ? DEFAULT_REALTIME_TRACKING_MODE_URL
            : realtimeTrackingModeUrl.Trim();
    }

    private string ResolveRealtimeTrackingStatusUrl()
    {
        return string.IsNullOrWhiteSpace(realtimeTrackingStatusUrl)
            ? DEFAULT_REALTIME_TRACKING_STATUS_URL
            : realtimeTrackingStatusUrl.Trim();
    }

    void Start()
    {
        initialize = this;
        startup_session_id = BuildStartupSessionId();
        StartCoroutine(PlaceStartupCameraMarkerWhenReady());
        realtimeTrackingStatusPollingCoroutine = StartCoroutine(PollRealtimeTrackingStatus());
        objectTrackingBoxPollingCoroutine = StartCoroutine(PollLatestObjectTrackingBoxes());

        // Pose sampling is started explicitly by the capture flow.
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
    }

    private void LogHttpRequestEnd(string label, HTTPRequest request, HTTPResponse response)
    {
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
        GenerateRequestContext context = request != null
            ? request.Tag as GenerateRequestContext
            : null;
        return context != null ? context.purpose : "";
    }

    long GetRequestDisplayGeneration(HTTPRequest request)
    {
        GenerateRequestContext context = request != null
            ? request.Tag as GenerateRequestContext
            : null;
        return context != null ? context.displayGeneration : -1;
    }

    /// <summary>
    /// Upload a near-range AHAT capture.
    /// </summary>

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
                photoTimeUtc = DateTime.UtcNow.ToString("o"),
            });
        }

        if (frames.Count < MARKER_CAPTURE_MIN_SUCCESS)
        {
            Game_M.initialize.XianShi("shangchuan_mark_ERR_no_frames");
            isMarkerCaptureActive = false;
            yield break;
        }

        SendArucoBatchGenerateRequest(frames);
        isMarkerCaptureActive = false;
    }

    void SendArucoBatchGenerateRequest(List<MarkerCaptureFrame> frames)
    {
        string url = "http://10.40.1.122:7355/generate";
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Post, OnRequestFinished);
        request.Tag = new GenerateRequestContext
        {
            purpose = TASK_PURPOSE_ARUCO_REFERENCE,
            displayGeneration = localModelDisplayGeneration,
        };

        Game_M.initialize.XianShi("shangchuan_Dabao");
        request.AddField("purpose", TASK_PURPOSE_ARUCO_REFERENCE);

        JArray frameArray = new JArray();
        for (int i = 0; i < frames.Count; i++)
        {
            MarkerCaptureFrame frame = frames[i];
            JObject frameJ = new JObject
            {
                ["width"] = frame.pvWidth,
                ["height"] = frame.pvHeight,
                ["k"] = Float2DToJArray(frame.pvK),
                ["pose"] = Float2DToJArray(frame.pvPose),
                ["time"] = frame.photoTimeUtc,
            };
            frameArray.Add(frameJ);
            request.AddBinaryData("pv_image_" + i, frame.pvPng, "pv_" + i + ".png", "image/png");
        }
        request.AddField("PVCameraFramesJ", frameArray.ToString(Formatting.None));

        JObject deviceJ = new JObject
        {
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
        string ip,
        string photoTimeUtc,
        byte[] depthPng = null,
        float[,] depthPose = null,
        string sensorType = null,
        Vector2? boxTL = null,
        Vector2? boxBR = null,
        bool forceNew3dModel = false
    )
    {
        string url = "http://10.40.1.122:7355/generate";
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Post, OnRequestFinished);
        request.Tag = new GenerateRequestContext
        {
            purpose = purpose,
            displayGeneration = localModelDisplayGeneration,
        };

        Game_M.initialize.XianShi("shangchuan_Dabao");
        request.AddField("purpose", purpose);
        if (purpose == TASK_PURPOSE_OBJECT_RECONSTRUCTION)
        {
            request.AddField("force_new_3d_model", forceNew3dModel ? "1" : "0");
        }

        JObject PVCameraJ = new JObject
        {
            ["width"] = pvWidth,
            ["height"] = pvHeight,
            ["k"] = Float2DToJArray(pvK),
            ["pose"] = Float2DToJArray(pvPose),
            ["time"] = photoTimeUtc,
        };
        request.AddField("PVCameraJ", PVCameraJ.ToString(Formatting.None));
        request.AddBinaryData("pv_image", texPvPng, "pv.png", "image/png");

        JObject deviceJ = new JObject
        {
            ["ip"] = string.IsNullOrEmpty(ip) ? "" : ip,
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

        // Freeze one consistent PV/depth pair before encoding the upload.
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
        //request.AddHeader("Content-Type", "multipart/form-data");
        // Encode the frozen PV frame and its camera parameters.
        Game_M.initialize.XianShi("shangchuan_PV");
        yield return null;
        byte[] tex_pv_P_C_F = ImageConversion.EncodeToPNG(PV_controler.tex_pv_frozen);
        yield return null;
        ushort width_pv_C_F = PV_controler.width_pv_frozen;
        ushort height_pv_C_F = PV_controler.height_pv_frozen;
        float[,] k_pv_C_F = PV_controler.k_pv_frozen;
        float[,] pose_pv_C_F = PV_controler.pose_pv_frozen;


        // Encode the aligned frozen depth frame.
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
        // Ask the user to confirm the object selection box on the frozen PV frame.
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

        // The selection panel owns preview, interaction, and confirmation.
        yield return StartCoroutine(
            selectionPanelManager.RequestSelection(PV_controler.tex_pv_frozen, cam.transform)
        );

        Game_M.initialize.XianShi("select_box_03_after_startcoroutine");

        // Cancelled selections do not create a server task.
        if (!selectionPanelManager.LastConfirmed)
        {
            Game_M.initialize.XianShi("select_box_cancel");
            yield break;
        }

        // Read the confirmed normalized rectangle.
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
            ip,
            photoTimeUtc,
            image_dp_P_C_F,
            pose_dp_C_F,
            sensorType,
            boxTL,
            boxBR,
            selectionPanelManager.LastForceRebuild
        );
    }

    public string task_id;

    private void OnRequestFinished(HTTPRequest request, HTTPResponse response)
    {
        string requestPurpose = GetRequestPurpose(request);
        LogHttpRequestEnd("generate:" + requestPurpose, request, response);
        if (response != null && response.IsSuccess)
        {
            JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
            string returnedTaskId = ReadString(jo, "task_id");
            if (string.IsNullOrEmpty(returnedTaskId))
            {
                ShowFrontMessage("generate_ERR_missing_task_id");
                return;
            }

            task_id = returnedTaskId;

            EnqueueAsyncUpdateTask(
                returnedTaskId,
                requestPurpose,
                false,
                GetRequestDisplayGeneration(request));
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
        ClearRealtimeTrackingTransfersForLocalHide();

        HistoryPresentationController historyController =
            HistoryPresentationController.Instance;
        if (historyController != null)
        {
            historyController.ResumeAllPresentations(false);
        }
        ObjectEvidenceDisplay evidenceDisplay = ObjectEvidenceDisplay.Instance;
        if (evidenceDisplay != null)
        {
            evidenceDisplay.Clear();
        }

        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null)
        {
            ShowFrontMessage("runtime_model_mgr_missing");
            return;
        }

        int hiddenCount = manager.HideAllRuntimeModels();
        Debug.Log("[RuntimeModelManager] Hid retained runtime models without deleting records/files: count="
            + hiddenCount.ToString(CultureInfo.InvariantCulture));
        ShowFrontMessage("runtime_model_hide_" + hiddenCount.ToString(CultureInfo.InvariantCulture));
    }

    private void ClearRealtimeTrackingTransfersForLocalHide()
    {
        automaticModelDeliverySuppressed = true;
        realtimeTrackingTransportGeneration++;
        localModelDisplayGeneration++;
        CancelRealtimeTrackingStatusRequest();

        if (historyTrackingModeRequest != null)
        {
            HistoryTrackingModeRequestContext modeContext =
                historyTrackingModeRequest.Tag as HistoryTrackingModeRequestContext;
            historyTrackingModeRequest.Abort();
            historyTrackingModeRequest = null;
            if (modeContext != null)
            {
                historyTrackingMode = modeContext.previousStableMode;
            }
        }

        List<HTTPRequest> downloadRequests = new List<HTTPRequest>(realtimeTrackingModelDownloadRequests);
        realtimeTrackingModelDownloadRequests.Clear();
        foreach (HTTPRequest request in downloadRequests)
        {
            PendingModelDownload pendingDownload = request != null
                ? request.Tag as PendingModelDownload
                : null;
            if (pendingDownload != null)
            {
                pendingDownload.cancelledByLocalHide = true;
            }
            ReleaseRuntimeModelDownload(pendingDownload);
            if (request != null)
            {
                request.Abort();
            }
        }

        int pendingCount = pendingModelLoadQueue.Count;
        for (int i = 0; i < pendingCount; i++)
        {
            PendingModelDownload pendingDownload = pendingModelLoadQueue.Dequeue();
            if (pendingDownload != null && pendingDownload.isRealtimeTrackingDownload)
            {
                ReleaseRuntimeModelDownload(pendingDownload);
                continue;
            }
            pendingModelLoadQueue.Enqueue(pendingDownload);
        }

        if (activeModelLoad != null)
        {
            // TriLib cannot safely cancel midway; retain the staged record/file but keep it hidden.
            activeModelLoad.hideWhenComplete = true;
        }
    }

    private void EnqueueAsyncUpdateTask(string queuedTaskId, string purpose)
    {
        EnqueueAsyncUpdateTask(queuedTaskId, purpose, false, localModelDisplayGeneration);
    }

    private void EnqueueAsyncUpdateTask(string queuedTaskId, string purpose, bool insertAtFront)
    {
        EnqueueAsyncUpdateTask(queuedTaskId, purpose, insertAtFront, localModelDisplayGeneration);
    }

    private void EnqueueAsyncUpdateTask(
        string queuedTaskId,
        string purpose,
        bool insertAtFront,
        long displayGeneration)
    {
        if (string.IsNullOrEmpty(queuedTaskId)
            || (purpose != TASK_PURPOSE_OBJECT_RECONSTRUCTION
                && purpose != TASK_PURPOSE_ARUCO_REFERENCE))
        {
            Debug.LogWarning("[ASYNC_QUEUE] Reject task without a canonical purpose.");
            return;
        }

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
            purpose = purpose,
            displayGeneration = displayGeneration,
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
            ["startup_session_id"] = startup_session_id ?? "",
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

    private void ResumeAsyncTaskQueuePolling()
    {
        asyncTaskQueuePaused = false;
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

        string completedTaskId = ReadString(wrapper, "task_id");
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
        string purpose = ReadString(wrapper, "purpose");
        string status = ReadString(wrapper, "status");
        if (string.IsNullOrEmpty(purpose)
            || string.IsNullOrEmpty(status)
            || (!string.IsNullOrEmpty(queuedPurpose) && purpose != queuedPurpose)
            || ReadString(taskResponse, "task_id") != completedTaskId
            || ReadString(taskResponse, "status") != status
            || ReadString(taskResponse, "purpose") != purpose)
        {
            Debug.LogWarning("[ASYNC_QUEUE] Reject non-canonical task envelope.");
            string ignoredPurpose;
            RemoveAsyncUpdateTask(completedTaskId, out ignoredPurpose);
            EnsureAsyncTaskQueuePolling();
            return;
        }
        long taskDisplayGeneration = pendingEntry != null
            ? pendingEntry.displayGeneration
            : localModelDisplayGeneration;
        bool allowAutomaticModelDelivery = taskDisplayGeneration == localModelDisplayGeneration;

        RemoveAsyncUpdateTask(completedTaskId, out queuedPurpose);

        asyncTaskQueuePaused = true;
        task_id = completedTaskId;

        if (status == "failed" || status == "upload_failed")
        {
            string err = taskResponse["error"]?.ToString();
            Debug.LogError("[ASYNC_QUEUE] task failed: " + err);
            ShowFrontMessage(NormalizeServerErrorForFrontMessage(err, "check_ERR_task_failed", purpose));
            ResumeAsyncTaskQueuePolling();
            return;
        }

        if (purpose == TASK_PURPOSE_ARUCO_REFERENCE)
        {
            if (status != "aruco_completed")
            {
                Debug.LogWarning("[ASYNC_QUEUE] Reject invalid ArUco terminal status: " + status);
                ShowFrontMessage("check_ERR_unknown_terminal_status");
                ResumeAsyncTaskQueuePolling();
                return;
            }
            ApplyDebugInfo(taskResponse);
            bool arucoDetected = IsArucoDetected(taskResponse);
            startupArUcoReferenceReady = arucoDetected;
            if (startupArUcoReferenceReady)
            {
                RequestHistoryTrackingMode(false, true);
            }
            ShowFrontMessage(arucoDetected ? "aruco_completed" : "aruco_ERR_missing_reference");
            ResumeAsyncTaskQueuePolling();
            return;
        }

        if (purpose != TASK_PURPOSE_OBJECT_RECONSTRUCTION || status != "completed")
        {
            Debug.LogWarning("[ASYNC_QUEUE] unsupported ready status: " + status);
            ShowFrontMessage("check_ERR_unknown_terminal_status");
            ResumeAsyncTaskQueuePolling();
            return;
        }

        pendingModelShouldPlaceDebugMarkers = false;
        pendingModelDisplayGeneration = taskDisplayGeneration;
        if (!ApplyCompletedTaskResponse(taskResponse, "ASYNC_QUEUE"))
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

        if (!allowAutomaticModelDelivery)
        {
            Debug.Log("[ASYNC_QUEUE] Model remains on server/cache after local display clear: "
                + completedTaskId);
            ResumeAsyncTaskQueuePolling();
            return;
        }

        RuntimeModelManager completedManager = RuntimeModelManager.Instance;
        if (completedManager != null && completedManager.HasMatchingModel(pendingModelInstance))
        {
            Debug.Log("[ASYNC_QUEUE] completed model revision is already local.");
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

            string taskId = ReadString(pendingTask, "task_id");
            string queuedPurpose = FindQueuedPurpose(taskId);
            string purpose = ReadString(pendingTask, "purpose");
            if (!string.IsNullOrEmpty(queuedPurpose) && purpose != queuedPurpose)
            {
                continue;
            }
            if (purpose != TASK_PURPOSE_OBJECT_RECONSTRUCTION)
            {
                continue;
            }

            RuntimeModelInstance hintInstance;
            if (!TryBuildPendingSpatialHintInstance(pendingTask, out hintInstance))
            {
                continue;
            }

            ModelEventDisplay eventDisplay = ModelEventDisplay.Instance;
            if (eventDisplay != null)
            {
                eventDisplay.ShowForModel(hintInstance, BuildPendingProgressMessage(pendingTask));
            }
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
        string progressText = ReadString(pendingTask, "progress_text");
        return string.IsNullOrEmpty(progressText) ? "processing" : progressText;
    }

    private bool TryBuildPendingSpatialHintInstance(JObject pendingTask, out RuntimeModelInstance instance)
    {
        instance = null;
        if (pendingTask == null)
        {
            return false;
        }

        RuntimeSpatialBoxData spatialBox;
        if (!TryParsePendingSam3SpatialBoxToken(
                pendingTask["sam3_spatial_box"],
                out spatialBox))
        {
            return false;
        }

        string taskId = pendingTask["task_id"]?.ToString().Trim() ?? "";
        if (string.IsNullOrEmpty(taskId))
        {
            return false;
        }

        instance = new RuntimeModelInstance
        {
            ModelKey = taskId,
            TaskId = taskId,
            FbxUrl = "",
            Pose = new RuntimeModelPoseData(),
            SpatialBox = spatialBox,
        };
        return true;
    }

    public void ToggleHistoryTrackingMode()
    {
        HistoryPresentationController controller =
            HistoryPresentationController.Instance;
        if (controller == null)
        {
            ShowFrontMessage("history_presentation_controller_missing");
            return;
        }
        controller.ToggleAllPresentations();
    }

    public void ShowLatestHistoryPlacements()
    {
        HistoryPresentationController controller =
            HistoryPresentationController.Instance;
        if (controller == null)
        {
            ShowFrontMessage("history_presentation_controller_missing");
            return;
        }
        controller.ShowLatestPlacements();
    }

    private void RequestHistoryTrackingMode(bool historyModeEnabled, bool isAutomaticRehandshake)
    {
        if (historyModeEnabled)
        {
            Debug.LogError(
                "[HISTORY_TRACKING] Server history mode was retired; "
                + "history is a local presentation state only.");
            return;
        }

        HistoryTrackingMode targetMode = historyModeEnabled
            ? HistoryTrackingMode.History
            : HistoryTrackingMode.Live;
        if (!isAutomaticRehandshake
            && historyTrackingMode == targetMode
            && historyTrackingModeRequest == null
            && !(targetMode == HistoryTrackingMode.Live
                && (!realtimeTrackingStatusDeliveryEnabled || automaticModelDeliverySuppressed)))
        {
            ShowFrontMessage(historyModeEnabled ? "history_tracking_history" : "history_tracking_live");
            return;
        }

        HistoryTrackingMode previousStableMode = ResolvePreviousStableHistoryTrackingMode();
        CancelRealtimeTrackingStatusRequest();
        acceptedHistoryTrackingModeEpoch = -1;
        historyTrackingRequestGeneration++;
        long requestGeneration = historyTrackingRequestGeneration;
        if (historyTrackingModeRequest != null)
        {
            historyTrackingModeRequest.Abort();
            historyTrackingModeRequest = null;
        }

        historyTrackingMode = historyModeEnabled
            ? HistoryTrackingMode.EnteringHistory
            : HistoryTrackingMode.EnteringLive;
        JObject payload = new JObject
        {
            ["startup_session_id"] = startup_session_id,
            ["mode"] = historyModeEnabled ? "history" : "live",
            ["request_generation"] = requestGeneration,
        };
        HistoryTrackingModeRequestContext context = new HistoryTrackingModeRequestContext
        {
            requestGeneration = requestGeneration,
            targetMode = targetMode,
            previousStableMode = previousStableMode,
            isAutomaticRehandshake = isAutomaticRehandshake,
        };

        string url = ResolveRealtimeTrackingModeUrl();
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Post, OnHistoryTrackingModeFinished);
        request.Tag = context;
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.RawData = Encoding.UTF8.GetBytes(payload.ToString(Formatting.None));
        historyTrackingModeRequest = request;
        LogHttpRequestStart("realtime-tracking/mode", request);
        request.Send();
        if (!isAutomaticRehandshake)
        {
            ShowFrontMessage(historyModeEnabled
                ? "history_tracking_entering_history"
                : "history_tracking_entering_live");
        }
    }

    private HistoryTrackingMode ResolvePreviousStableHistoryTrackingMode()
    {
        return historyTrackingMode == HistoryTrackingMode.History
            || historyTrackingMode == HistoryTrackingMode.EnteringLive
            ? HistoryTrackingMode.History
            : HistoryTrackingMode.Live;
    }

    private IEnumerator PollLatestObjectTrackingBoxes()
    {
        while (true)
        {
            if (objectTrackingBoxRequest == null
                && !string.IsNullOrEmpty(startup_session_id))
            {
                RequestLatestObjectTrackingBoxes();
            }
            yield return new WaitForSecondsRealtime(
                SHIGURE_BOX_POLL_INTERVAL_SECONDS);
        }
    }

    private void RequestLatestObjectTrackingBoxes()
    {
        if (objectTrackingBoxRequest != null
            || string.IsNullOrEmpty(startup_session_id)
            || !TryGetServerServiceBaseUri(out Uri serviceBaseUri))
        {
            return;
        }
        Uri endpoint = new Uri(
            serviceBaseUri,
            "api/v2/shigure/object-tracking-boxes/latest"
                + "?startup_session_id="
                + Uri.EscapeDataString(startup_session_id));
        HTTPRequest request = new HTTPRequest(
            endpoint,
            HTTPMethods.Get,
            OnLatestObjectTrackingBoxesFinished);
        request.ConnectTimeout = TimeSpan.FromSeconds(2);
        request.Timeout = TimeSpan.FromSeconds(5);
        request.AddHeader("Accept", "application/json");
        objectTrackingBoxRequest = request;
        request.Send();
    }

    private void OnLatestObjectTrackingBoxesFinished(
        HTTPRequest request,
        HTTPResponse response)
    {
        if (request != objectTrackingBoxRequest)
        {
            return;
        }
        objectTrackingBoxRequest = null;
        if (response == null || !response.IsSuccess)
        {
            Debug.LogWarning("[SHIGURE_BOX] latest relay request failed; keep current boxes.");
            return;
        }

        JObject root;
        try
        {
            root = JObject.Parse(response.DataAsText);
        }
        catch (Exception exc)
        {
            Debug.LogWarning("[SHIGURE_BOX] invalid relay JSON: " + exc.Message);
            return;
        }
        if (root["success"] == null
            || !root["success"].Value<bool>()
            || ReadString(root, "startup_session_id") != startup_session_id)
        {
            return;
        }

        string coordinateEpoch = ReadString(root, "coordinate_epoch");
        JArray items = root["tracking_boxes"] as JArray ?? new JArray();
        Dictionary<string, RuntimeSpatialBoxData> latest =
            new Dictionary<string, RuntimeSpatialBoxData>(StringComparer.Ordinal);
        foreach (JToken token in items)
        {
            JObject item = token as JObject;
            string trackingId = ReadString(item, "tracking_id");
            if (item == null
                || string.IsNullOrEmpty(trackingId)
                || !TryParseCurrentSpatialBoxToken(
                    item["spatial_box"],
                    out RuntimeSpatialBoxData spatialBox,
                    out bool noBox,
                    out long revision)
                || noBox
                || spatialBox == null)
            {
                continue;
            }
            spatialBox.Revision = revision;
            latest[trackingId] = spatialBox;
        }

        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager != null)
        {
            manager.ReplaceRawTrackingBoxSnapshot(latest, coordinateEpoch);
        }
        Debug.Log(
            "[SHIGURE_BOX] relayed latest snapshot count="
            + latest.Count.ToString(CultureInfo.InvariantCulture));
    }

    private void CancelLatestObjectTrackingBoxRequest()
    {
        HTTPRequest request = objectTrackingBoxRequest;
        objectTrackingBoxRequest = null;
        if (request != null)
        {
            request.Abort();
        }
    }

    private IEnumerator PollRealtimeTrackingStatus()
    {
        while (true)
        {
            if (historyTrackingMode == HistoryTrackingMode.Live
                && realtimeTrackingStatusDeliveryEnabled
                && historyTrackingModeRequest == null
                && realtimeTrackingStatusRequest == null
                && !string.IsNullOrEmpty(startup_session_id))
            {
                if (acceptedHistoryTrackingModeEpoch >= 0)
                {
                    RequestRealtimeTrackingStatus();
                }
                else if (startupArUcoReferenceReady)
                {
                    RequestHistoryTrackingMode(false, true);
                }
            }
            yield return new WaitForSecondsRealtime(
                Mathf.Max(0.25f, realtimeTrackingStatusPollIntervalSeconds));
        }
    }

    private void RequestRealtimeTrackingStatus()
    {
        if (historyTrackingMode != HistoryTrackingMode.Live
            || !realtimeTrackingStatusDeliveryEnabled
            || historyTrackingModeRequest != null
            || realtimeTrackingStatusRequest != null
            || string.IsNullOrEmpty(startup_session_id)
            || acceptedHistoryTrackingModeEpoch < 0)
        {
            return;
        }

        string baseUrl = ResolveRealtimeTrackingStatusUrl();
        string separator = baseUrl.IndexOf('?') >= 0 ? "&" : "?";
        string url = baseUrl + separator + "startup_session_id="
            + Uri.EscapeDataString(startup_session_id);
        RealtimeTrackingStatusRequestContext context = new RealtimeTrackingStatusRequestContext
        {
            transportGeneration = realtimeTrackingTransportGeneration,
            requestGeneration = historyTrackingRequestGeneration,
            modeEpoch = acceptedHistoryTrackingModeEpoch,
        };
        var request = new HTTPRequest(new Uri(url), HTTPMethods.Get, OnRealtimeTrackingStatusFinished);
        request.Tag = context;
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        realtimeTrackingStatusRequest = request;
        LogHttpRequestStart("realtime-tracking/status", request);
        request.Send();
    }

    private void CancelRealtimeTrackingStatusRequest()
    {
        HTTPRequest request = realtimeTrackingStatusRequest;
        realtimeTrackingStatusRequest = null;
        if (request != null)
        {
            request.Abort();
        }
    }

    private void BeginAutomaticLiveRehandshake()
    {
        if (historyTrackingMode != HistoryTrackingMode.Live || historyTrackingModeRequest != null)
        {
            return;
        }
        Debug.Log("[HISTORY_TRACKING] Re-handshake Live mode after server state changed.");
        RequestHistoryTrackingMode(false, true);
    }

    private void OnRealtimeTrackingStatusFinished(HTTPRequest request, HTTPResponse response)
    {
        LogHttpRequestEnd("realtime-tracking/status", request, response);
        RealtimeTrackingStatusRequestContext context = request != null
            ? request.Tag as RealtimeTrackingStatusRequestContext
            : null;
        if (request != realtimeTrackingStatusRequest)
        {
            return;
        }
        realtimeTrackingStatusRequest = null;

        if (context == null
            || context.transportGeneration != realtimeTrackingTransportGeneration
            || historyTrackingMode != HistoryTrackingMode.Live
            || historyTrackingModeRequest != null
            || context.requestGeneration != historyTrackingRequestGeneration)
        {
            return;
        }
        if (response == null || !response.IsSuccess)
        {
            Debug.LogWarning("[HISTORY_TRACKING] status request failed.");
            BeginAutomaticLiveRehandshake();
            return;
        }

        JObject root;
        try
        {
            root = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        }
        catch (Exception exc)
        {
            Debug.LogWarning("[HISTORY_TRACKING] invalid status response: " + exc.Message);
            return;
        }

        if (!ValidateTrackingResponseHeader(
                root,
                HistoryTrackingMode.Live,
                out long modeEpoch,
                out string coordinateEpoch))
        {
            if (root != null
                && ReadString(root, "startup_session_id") == startup_session_id
                && ReadString(root, "mode") == "live")
            {
                BeginAutomaticLiveRehandshake();
            }
            return;
        }
        if (modeEpoch != context.modeEpoch
            || ReadLong(root, "request_generation") != context.requestGeneration)
        {
            BeginAutomaticLiveRehandshake();
            return;
        }

        if (!ApplyTrackingSnapshot(
                root,
                modeEpoch,
                coordinateEpoch,
                out int appliedCount))
        {
            Debug.LogWarning(
                "[HISTORY_TRACKING] Rejected status snapshot before applying "
                + "pose, box, or model-download state.");
            return;
        }
        QueueMissingRealtimeTrackingModels(root);
    }

    private void OnDestroy()
    {
        CancelRealtimeTrackingStatusRequest();
        CancelLatestObjectTrackingBoxRequest();
        realtimeTrackingStatusPollingCoroutine = null;
        objectTrackingBoxPollingCoroutine = null;
    }

    private void OnHistoryTrackingModeFinished(HTTPRequest request, HTTPResponse response)
    {
        LogHttpRequestEnd("realtime-tracking/mode", request, response);
        HistoryTrackingModeRequestContext context = request != null
            ? request.Tag as HistoryTrackingModeRequestContext
            : null;
        if (context == null
            || context.requestGeneration != historyTrackingRequestGeneration
            || request != historyTrackingModeRequest)
        {
            return;
        }
        historyTrackingModeRequest = null;

        if (response == null || !response.IsSuccess)
        {
            historyTrackingMode = context.previousStableMode;
            ShowFrontMessage("history_tracking_ERR_request_failed");
            return;
        }

        JObject root;
        try
        {
            root = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        }
        catch (Exception exc)
        {
            historyTrackingMode = context.previousStableMode;
            Debug.LogError("[HISTORY_TRACKING] invalid mode response: " + exc.Message);
            ShowFrontMessage("history_tracking_ERR_invalid_response");
            return;
        }

        if (!ValidateTrackingResponseHeader(
                root,
                context.targetMode,
                out long modeEpoch,
                out string coordinateEpoch)
            || ReadLong(root, "request_generation") != context.requestGeneration)
        {
            historyTrackingMode = context.previousStableMode;
            ShowFrontMessage("history_tracking_ERR_invalid_response");
            return;
        }

        if (!ApplyTrackingSnapshot(
                root,
                modeEpoch,
                coordinateEpoch,
                out int appliedCount))
        {
            historyTrackingMode = context.previousStableMode;
            Debug.LogWarning(
                "[HISTORY_TRACKING] Rejected mode snapshot before applying "
                + "pose, box, or model-download state.");
            ShowFrontMessage("history_tracking_ERR_invalid_response");
            return;
        }

        acceptedHistoryTrackingModeEpoch = modeEpoch;
        historyTrackingMode = context.targetMode;
        realtimeTrackingStatusDeliveryEnabled = context.targetMode == HistoryTrackingMode.Live;
        if (!context.isAutomaticRehandshake)
        {
            automaticModelDeliverySuppressed = false;
            RuntimeModelManager manager = RuntimeModelManager.Instance;
            if (manager != null)
            {
                manager.ShowAllRuntimeModels();
            }
        }

        int queuedModelCount = QueueMissingRealtimeTrackingModels(root);
        Debug.Log(
            "[HISTORY_TRACKING] mode=" + ReadString(root, "mode")
            + " generation=" + context.requestGeneration.ToString(CultureInfo.InvariantCulture)
            + " mode_epoch=" + modeEpoch.ToString(CultureInfo.InvariantCulture)
            + " coordinate_epoch=" + coordinateEpoch
            + " applied=" + appliedCount.ToString(CultureInfo.InvariantCulture)
            + " model_downloads=" + queuedModelCount.ToString(CultureInfo.InvariantCulture));

        if (!context.isAutomaticRehandshake)
        {
            ShowFrontMessage(
                context.targetMode == HistoryTrackingMode.History
                    ? "history_tracking_history_" + appliedCount.ToString(CultureInfo.InvariantCulture)
                    : "history_tracking_live_" + appliedCount.ToString(CultureInfo.InvariantCulture));
        }
    }

    private bool ValidateTrackingResponseHeader(
        JObject root,
        HistoryTrackingMode expectedMode,
        out long modeEpoch,
        out string coordinateEpoch)
    {
        modeEpoch = -1;
        coordinateEpoch = "";
        if (root == null
            || root["success"] == null
            || root["success"].Type != JTokenType.Boolean
            || !root["success"].Value<bool>()
            || ReadString(root, "startup_session_id") != startup_session_id
            || !ServerModeMatchesTarget(ReadString(root, "mode"), expectedMode))
        {
            return false;
        }

        modeEpoch = ReadLong(root, "mode_epoch");
        coordinateEpoch = ReadString(root, "coordinate_epoch");
        return modeEpoch >= 0;
    }

    private bool TryValidateTrackingSnapshot(
        JObject response,
        out JArray snapshotItems)
    {
        snapshotItems = response != null
            ? response["items"] as JArray
            : null;
        long declaredItemCount = ReadLong(response, "count");
        if (snapshotItems == null
            || declaredItemCount < 0
            || declaredItemCount > 5
            || declaredItemCount != snapshotItems.Count)
        {
            Debug.LogWarning(
                "[HISTORY_TRACKING] Ignore incomplete tracking snapshot; "
                + "no pose, box, or model download was changed.");
            return false;
        }

        HashSet<string> validatedPoseDisplayObjectIds =
            new HashSet<string>(StringComparer.Ordinal);
        foreach (JToken token in snapshotItems)
        {
            JObject item = token as JObject;
            string displayObjectId = ReadString(item, "display_object_id");
            long modelRevision = ReadLong(item, "model_revision");
            string poseSource = ReadString(item, "pose_source");
            long poseRevision = ReadLong(item, "pose_revision");
            string coordinateSpace = ReadString(item, "coordinate_space");
            JObject poseObject = item != null
                ? item["pose"] as JObject
                : null;
            if (item == null
                || string.IsNullOrEmpty(displayObjectId)
                || !validatedPoseDisplayObjectIds.Add(displayObjectId)
                || modelRevision <= 0
                || poseRevision <= 0
                || (poseSource != "hololens" && poseSource != "tracking")
                || coordinateSpace != "hololens_current_local"
                || !HasExactKeys(
                    poseObject,
                    "position",
                    "rotation_quaternion_xyzw")
                || !TryParsePoseToken(
                    poseObject,
                    out Vector3 ignoredPosition,
                    out Quaternion ignoredRotation))
            {
                Debug.LogWarning(
                    "[HISTORY_TRACKING] Ignore malformed tracking item; "
                    + "no pose, box, or model download was changed.");
                return false;
            }

            JProperty modelProperty = item.Property("model");
            if (modelProperty != null
                && !TryValidateCanonicalRealtimeModelPayload(
                    item,
                    modelProperty.Value as JObject))
            {
                Debug.LogWarning(
                    "[HISTORY_TRACKING] Reject non-canonical model payload "
                    + "before applying the tracking snapshot.");
                return false;
            }
        }

        return true;
    }

    private bool TryValidateCanonicalRealtimeModelPayload(
        JObject item,
        JObject model)
    {
        if (item == null || model == null)
        {
            return false;
        }

        string displayObjectId = ReadString(item, "display_object_id");
        long modelRevision = ReadLong(item, "model_revision");
        string activeTaskId = ReadString(item, "active_model_task_id");
        string taskId = ReadString(model, "task_id");
        string modelKey = ReadString(model, "model_key");
        string fbxUrl = ReadString(model, "fbx_url");
        JObject pose = item["pose"] as JObject;
        bool validHttpUrl = Uri.TryCreate(
                fbxUrl,
                UriKind.Absolute,
                out Uri parsedFbxUri)
            && (parsedFbxUri.Scheme == Uri.UriSchemeHttp
                || parsedFbxUri.Scheme == Uri.UriSchemeHttps);
        return HasExactKeys(
                model,
                "model_key",
                "task_id",
                "display_object_id",
                "model_revision",
                "fbx_url",
                "pose",
                "coordinate_space")
            && !string.IsNullOrEmpty(displayObjectId)
            && modelRevision > 0
            && !string.IsNullOrEmpty(activeTaskId)
            && !string.IsNullOrEmpty(taskId)
            && modelKey == taskId
            && taskId == activeTaskId
            && validHttpUrl
            && ReadString(item, "coordinate_space")
                == "hololens_current_local"
            && ReadString(model, "display_object_id") == displayObjectId
            && ReadLong(model, "model_revision") == modelRevision
            && ReadString(model, "coordinate_space")
                == "hololens_current_local"
            && JToken.DeepEquals(model["pose"], pose);
    }

    private bool ApplyTrackingSnapshot(
        JObject response,
        long modeEpoch,
        string coordinateEpoch,
        out int appliedCount)
    {
        appliedCount = 0;
        if (string.IsNullOrEmpty(coordinateEpoch)
            || !TryValidateTrackingSnapshot(
                response,
                out JArray snapshotItems))
        {
            return false;
        }

        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null)
        {
            return false;
        }

        int resumedHistoryCount =
            manager.ResumeHistoryPresentationsForCoordinateEpoch(
                coordinateEpoch);
        if (resumedHistoryCount > 0)
        {
            Debug.Log(
                "[HISTORY_TRACKING] Coordinate epoch changed to "
                + coordinateEpoch
                + "; resumed "
                + resumedHistoryCount.ToString(CultureInfo.InvariantCulture)
                + " history presentation(s).");
        }

        foreach (JToken token in snapshotItems)
        {
            JObject item = (JObject)token;
            string displayObjectId = ReadString(item, "display_object_id");
            long modelRevision = ReadLong(item, "model_revision");
            string poseSource = ReadString(item, "pose_source");
            long poseRevision = ReadLong(item, "pose_revision");
            TryParsePoseToken(
                item["pose"],
                out Vector3 position,
                out Quaternion rotation);

            RuntimeModelPoseData pose = new RuntimeModelPoseData
            {
                HasHololensPose = true,
                HololensPosition = position,
                HololensRotation = rotation,
            };
            if (manager.UpdateDisplayObjectPose(
                displayObjectId,
                modelRevision,
                poseSource,
                poseRevision,
                modeEpoch,
                coordinateEpoch,
                pose,
                out string rejectionReason))
            {
                appliedCount++;
            }
            else if (rejectionReason != "display_object_model_not_loaded"
                && rejectionReason != "pose_revision_duplicate")
            {
                Debug.Log("[HISTORY_TRACKING] Pose rejected for "
                    + displayObjectId + ": " + rejectionReason);
            }
        }
        return true;
    }

    private int QueueMissingRealtimeTrackingModels(JObject response)
    {
        if (automaticModelDeliverySuppressed)
        {
            return 0;
        }
        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null)
        {
            return 0;
        }

        int queuedCount = 0;
        foreach (JObject item in EnumerateTrackingItems(response))
        {
            JObject model = item["model"] as JObject;
            if (model == null)
            {
                continue;
            }

            string displayObjectId = ReadString(item, "display_object_id");
            long modelRevision = ReadLong(item, "model_revision");
            if (!TryValidateCanonicalRealtimeModelPayload(item, model))
            {
                Debug.LogError(
                    "[HISTORY_TRACKING] Validated snapshot lost its canonical "
                    + "model invariant; skip download.");
                continue;
            }
            if (manager.TryGetLoadedRecordByDisplayObjectId(
                    displayObjectId,
                    out RuntimeModelRecord loaded)
                && loaded != null
                && loaded.ModelRevision >= modelRevision)
            {
                continue;
            }

            string identityKey = "display:" + displayObjectId
                + ":revision:" + modelRevision.ToString(CultureInfo.InvariantCulture);
            if (runtimeModelDownloadsInFlight.Contains(identityKey))
            {
                continue;
            }
            if (QueueRuntimeModelDownload(model, false, identityKey))
            {
                queuedCount++;
            }
        }
        return queuedCount;
    }

    private IEnumerable<JObject> EnumerateTrackingItems(JObject response)
    {
        JArray items = response != null ? response["items"] as JArray : null;
        if (items == null)
        {
            yield break;
        }
        foreach (JToken token in items)
        {
            if (token is JObject item)
            {
                yield return item;
            }
        }
    }

    private bool ServerModeMatchesTarget(string serverMode, HistoryTrackingMode targetMode)
    {
        return targetMode == HistoryTrackingMode.History
            ? serverMode == "history"
            : serverMode == "live";
    }

    private static string ReadString(JObject payload, string key)
    {
        JToken token = payload != null ? payload[key] : null;
        return token != null && token.Type == JTokenType.String
            ? token.Value<string>().Trim()
            : "";
    }

    private static long ReadLong(JObject payload, string key)
    {
        JToken token = payload != null ? payload[key] : null;
        return token != null && token.Type == JTokenType.Integer
            ? token.Value<long>()
            : -1;
    }

    private static bool HasExactKeys(JObject payload, params string[] expectedKeys)
    {
        if (payload == null || payload.Count != expectedKeys.Length)
        {
            return false;
        }
        HashSet<string> expected = new HashSet<string>(expectedKeys, StringComparer.Ordinal);
        foreach (JProperty property in payload.Properties())
        {
            if (!expected.Remove(property.Name))
            {
                return false;
            }
        }
        return expected.Count == 0;
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
        if (arr == null
            || arr.Count != 3
            || !IsJsonNumber(arr[0])
            || !IsJsonNumber(arr[1])
            || !IsJsonNumber(arr[2]))
        {
            return false;
        }

        try
        {
            float x = arr[0].Value<float>();
            float y = arr[1].Value<float>();
            float z = arr[2].Value<float>();
            if (!IsFinite(x) || !IsFinite(y) || !IsFinite(z))
            {
                return false;
            }
            value = new Vector3(x, y, z);
            return true;
        }
        catch (Exception)
        {
            value = Vector3.zero;
            return false;
        }
    }

    bool TryReadQuaternion(JToken token, out Quaternion value)
    {
        value = Quaternion.identity;
        JArray arr = token as JArray;
        if (arr == null
            || arr.Count != 4
            || !IsJsonNumber(arr[0])
            || !IsJsonNumber(arr[1])
            || !IsJsonNumber(arr[2])
            || !IsJsonNumber(arr[3]))
        {
            return false;
        }

        try
        {
            float x = arr[0].Value<float>();
            float y = arr[1].Value<float>();
            float z = arr[2].Value<float>();
            float w = arr[3].Value<float>();
            if (!IsFinite(x)
                || !IsFinite(y)
                || !IsFinite(z)
                || !IsFinite(w))
            {
                return false;
            }

            double sqrMagnitude =
                ((double)x * x)
                + ((double)y * y)
                + ((double)z * z)
                + ((double)w * w);
            if (double.IsNaN(sqrMagnitude)
                || double.IsInfinity(sqrMagnitude)
                || sqrMagnitude <= 1e-12)
            {
                return false;
            }

            float inverseMagnitude =
                (float)(1.0 / Math.Sqrt(sqrMagnitude));
            value = new Quaternion(
                x * inverseMagnitude,
                y * inverseMagnitude,
                z * inverseMagnitude,
                w * inverseMagnitude);
            return IsFinite(value.x)
                && IsFinite(value.y)
                && IsFinite(value.z)
                && IsFinite(value.w);
        }
        catch (Exception)
        {
            value = Quaternion.identity;
            return false;
        }
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
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

    private static bool IsJsonNumber(JToken token)
    {
        return token != null
            && (token.Type == JTokenType.Integer || token.Type == JTokenType.Float);
    }

    private bool TryParsePendingSam3SpatialBoxToken(
        JToken boxToken,
        out RuntimeSpatialBoxData spatialBox)
    {
        spatialBox = null;
        JObject box = boxToken as JObject;
        if (box == null
            || ReadString(box, "status") != "ready"
            || ReadString(box, "coordinate_space") != "unity_world"
            || !TryReadVector3(box["aabb_min_world"], out Vector3 minimum)
            || !TryReadVector3(box["aabb_max_world"], out Vector3 maximum))
        {
            return false;
        }

        Vector3 min = Vector3.Min(minimum, maximum);
        Vector3 max = Vector3.Max(minimum, maximum);
        spatialBox = new RuntimeSpatialBoxData
        {
            IsReady = true,
            Status = "ready",
            // The legacy pending preview already contains Unity/HoloLens-local
            // world coordinates. Normalize it at this isolated boundary; this
            // parser is never used by v2 live/history transport.
            CoordinateSpace = "hololens_current_local",
            Revision = 1,
            CornersWorld = new[]
            {
                new Vector3(min.x, min.y, min.z),
                new Vector3(max.x, min.y, min.z),
                new Vector3(max.x, max.y, min.z),
                new Vector3(min.x, max.y, min.z),
                new Vector3(min.x, min.y, max.z),
                new Vector3(max.x, min.y, max.z),
                new Vector3(max.x, max.y, max.z),
                new Vector3(min.x, max.y, max.z),
            },
        };
        return true;
    }

    private bool TryParseCurrentSpatialBoxToken(
        JToken boxToken,
        out RuntimeSpatialBoxData spatialBox,
        out bool noBox,
        out long revision)
    {
        spatialBox = null;
        noBox = false;
        revision = -1;
        if (boxToken == null || boxToken.Type == JTokenType.Null)
        {
            return false;
        }

        JObject box = boxToken as JObject;
        if (box == null)
        {
            return false;
        }
        string status = ReadString(box, "status");
        string coordinateSpace = ReadString(box, "coordinate_space");
        revision = ReadLong(box, "revision");
        if (status == "no_box")
        {
            if (!HasExactKeys(box, "status", "coordinate_space", "revision")
                || coordinateSpace != "hololens_current_local"
                || revision <= 0)
            {
                return false;
            }
            noBox = true;
            return true;
        }
        if (!HasExactKeys(
                box,
                "status",
                "coordinate_space",
                "revision",
                "corners_hololens_current_local_m")
            || status != "ready"
            || coordinateSpace != "hololens_current_local")
        {
            return false;
        }

        JArray corners =
            box["corners_hololens_current_local_m"] as JArray;
        if (revision <= 0 || corners == null || corners.Count != 8)
        {
            return false;
        }

        Vector3[] parsedCorners = new Vector3[8];
        for (int i = 0; i < parsedCorners.Length; i++)
        {
            if (!TryReadVector3(corners[i], out parsedCorners[i]))
            {
                return false;
            }
        }

        spatialBox = new RuntimeSpatialBoxData
        {
            IsReady = true,
            Status = status,
            CoordinateSpace = coordinateSpace,
            Revision = revision,
            CornersWorld = parsedCorners,
        };
        return true;
    }

    bool IsArucoDetected(JObject response)
    {
        JToken detected = response != null ? response["aruco_detected"] : null;
        return detected != null
            && detected.Type == JTokenType.Boolean
            && detected.Value<bool>();
    }

    bool TryBuildRuntimeModelInstance(
        JObject model,
        bool isEvidenceOverlay,
        out RuntimeModelInstance instance,
        out string errorMessage)
    {
        instance = null;
        errorMessage = "";
        if (model == null)
        {
            errorMessage = "download_ERR_missing_model_instance";
            return false;
        }

        string modelKey = ReadString(model, "model_key");
        string taskId = ReadString(model, "task_id");
        string displayObjectId = ReadString(model, "display_object_id");
        string fbxUrl = ReadString(model, "fbx_url");
        long modelRevision = ReadLong(model, "model_revision");
        string coordinateSpace = ReadString(model, "coordinate_space");
        JObject poseObject = model["pose"] as JObject;
        if (!HasExactKeys(
                model,
                "model_key",
                "task_id",
                "display_object_id",
                "model_revision",
                "fbx_url",
                "pose",
                "coordinate_space")
            || string.IsNullOrEmpty(modelKey)
            || string.IsNullOrEmpty(taskId)
            || string.IsNullOrEmpty(displayObjectId)
            || string.IsNullOrEmpty(fbxUrl)
            || modelRevision <= 0
            || coordinateSpace != "hololens_current_local"
            || !TryParsePoseToken(poseObject, out Vector3 position, out Quaternion rotation))
        {
            errorMessage = "download_ERR_invalid_model_contract";
            return false;
        }

        instance = new RuntimeModelInstance
        {
            ModelKey = modelKey,
            TaskId = taskId,
            FbxUrl = fbxUrl,
            DisplayObjectId = displayObjectId,
            ModelRevision = modelRevision,
            IsEvidenceOverlay = isEvidenceOverlay,
            Pose = new RuntimeModelPoseData
            {
                HasHololensPose = true,
                HololensPosition = position,
                HololensRotation = rotation,
            },
        };
        return true;
    }

    void ApplyResponsePoses(JObject response, RuntimeModelInstance instance)
    {
        if (instance != null && instance.Pose != null && instance.Pose.HasHololensPose)
        {
            serverObjectPosition = instance.Pose.HololensPosition;
            serverObjectRotation = instance.Pose.HololensRotation;
            hasServerPose = true;
        }
        else
        {
            hasServerPose = false;
        }

        JToken cameraPose = response?["debug"]?["pose_transform_stages"]?["pose_stage"]?["pv_camera_world"]?["pose"];
        if (TryParsePoseToken(cameraPose, out Vector3 cameraPosition, out Quaternion cameraRotation))
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

    bool ApplyCompletedTaskResponse(JObject response, string sourceTag)
    {
        JObject model = response != null ? response["model_instance"] as JObject : null;
        if (!TryBuildRuntimeModelInstance(
                model,
                false,
                out RuntimeModelInstance modelInstance,
                out string errorMessage))
        {
            Debug.LogWarning("[" + sourceTag + "] invalid canonical model_instance: " + errorMessage);
            ShowFrontMessage(errorMessage);
            return false;
        }

        pendingModelInstance = modelInstance;
        ApplyDebugInfo(response);
        ApplyResponsePoses(response, modelInstance);
        return true;
    }

    internal bool QueueRuntimeModelDownload(
        JObject model,
        bool isEvidenceOverlay,
        string explicitDownloadIdentityKey = "")
    {
        if (!TryBuildRuntimeModelInstance(
                model,
                isEvidenceOverlay,
                out RuntimeModelInstance instance,
                out string errorMessage))
        {
            ShowFrontMessage(errorMessage);
            return false;
        }

        pendingModelInstance = instance;
        pendingModelShouldPlaceDebugMarkers = false;
        pendingModelDisplayGeneration = localModelDisplayGeneration;
        task_id = instance.TaskId;
        if (TryQueueCachedRuntimeModelLoad(instance, explicitDownloadIdentityKey, false))
        {
            return true;
        }
        DownloadPendingRuntimeModel(false, explicitDownloadIdentityKey);
        return true;
    }

    private string BuildRuntimeModelDownloadIdentityKey(RuntimeModelInstance instance)
    {
        if (instance == null
            || string.IsNullOrEmpty(instance.DisplayObjectId)
            || instance.ModelRevision <= 0)
        {
            return "";
        }
        return (instance.IsEvidenceOverlay ? "evidence:" : "display:")
            + instance.DisplayObjectId
            + ":revision:"
            + instance.ModelRevision.ToString(CultureInfo.InvariantCulture);
    }

    private bool TryQueueCachedRuntimeModelLoad(
        RuntimeModelInstance instance,
        string explicitDownloadIdentityKey,
        bool resumeAsyncTaskQueueWhenComplete)
    {
        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null
            || instance == null
            || !manager.TryGetCachedModelPath(instance, out string cachedPath))
        {
            return false;
        }

        string downloadIdentityKey = string.IsNullOrEmpty(explicitDownloadIdentityKey)
            ? BuildRuntimeModelDownloadIdentityKey(instance)
            : explicitDownloadIdentityKey;
        if (!string.IsNullOrEmpty(downloadIdentityKey)
            && !runtimeModelDownloadsInFlight.Add(downloadIdentityKey))
        {
            return true;
        }

        PendingModelDownload pendingDownload = new PendingModelDownload
        {
            instance = instance,
            localPath = cachedPath,
            downloadIdentityKey = downloadIdentityKey,
            mayResumeAsyncTaskQueue = resumeAsyncTaskQueueWhenComplete,
            resumeAsyncTaskQueueWhenComplete = resumeAsyncTaskQueueWhenComplete && asyncTaskQueuePaused,
            isRealtimeTrackingDownload = !resumeAsyncTaskQueueWhenComplete,
            loadedFromCache = true,
            displayGeneration = pendingModelDisplayGeneration,
            showDebugMarkers = pendingModelShouldPlaceDebugMarkers,
            hasDebugCameraPose = hasServerCameraPose,
            debugCameraPosition = serverCameraPosition,
            debugCameraRotation = serverCameraRotation,
            hasDebugObjectPose = hasServerPose,
            debugObjectPosition = serverObjectPosition,
            debugObjectRotation = serverObjectRotation,
        };
        manager.PrepareForIncomingModel(instance);
        manager.ProtectCachedModelPath(cachedPath);
        EnqueueRuntimeModelLoad(pendingDownload);
        Debug.Log("[DOWNLOAD] Reuse cached runtime model: " + cachedPath);
        ShowFrontMessage("load_cached_model");
        return true;
    }

    /// <summary>
    /// Download or reuse the canonical runtime model and enqueue it for loading.
    /// </summary>
    private void DownloadPendingRuntimeModel(
        bool resumeAsyncTaskQueueWhenComplete = true,
        string explicitDownloadIdentityKey = "")
    {
        if (pendingModelInstance == null || string.IsNullOrEmpty(pendingModelInstance.FbxUrl))
        {
            ShowFrontMessage("download_ERR_missing_model_instance");
            if (resumeAsyncTaskQueueWhenComplete && asyncTaskQueuePaused)
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
            if (resumeAsyncTaskQueueWhenComplete && asyncTaskQueuePaused)
            {
                ResumeAsyncTaskQueuePolling();
            }
            return;
        }

        string downloadIdentityKey = string.IsNullOrEmpty(explicitDownloadIdentityKey)
            ? BuildRuntimeModelDownloadIdentityKey(pendingModelInstance)
            : explicitDownloadIdentityKey;
        if (!string.IsNullOrEmpty(downloadIdentityKey)
            && !runtimeModelDownloadsInFlight.Add(downloadIdentityKey))
        {
            Debug.Log("[DOWNLOAD] Skip duplicate in-flight model: " + downloadIdentityKey);
            if (resumeAsyncTaskQueueWhenComplete && asyncTaskQueuePaused)
            {
                ResumeAsyncTaskQueuePolling();
            }
            return;
        }

        manager.PrepareForIncomingModel(pendingModelInstance);

        PendingModelDownload pendingDownload = new PendingModelDownload
        {
            instance = pendingModelInstance,
            localPath = manager.CreateStableModelPath(pendingModelInstance),
            downloadIdentityKey = downloadIdentityKey,
            mayResumeAsyncTaskQueue = resumeAsyncTaskQueueWhenComplete,
            resumeAsyncTaskQueueWhenComplete = resumeAsyncTaskQueueWhenComplete && asyncTaskQueuePaused,
            isRealtimeTrackingDownload = !resumeAsyncTaskQueueWhenComplete,
            displayGeneration = pendingModelDisplayGeneration,
            showDebugMarkers = pendingModelShouldPlaceDebugMarkers,
            hasDebugCameraPose = hasServerCameraPose,
            debugCameraPosition = serverCameraPosition,
            debugCameraRotation = serverCameraRotation,
            hasDebugObjectPose = hasServerPose,
            debugObjectPosition = serverObjectPosition,
            debugObjectRotation = serverObjectRotation,
        };
        manager.ProtectCachedModelPath(pendingDownload.localPath);

        var request = new HTTPRequest(new Uri(pendingModelInstance.FbxUrl), HTTPMethods.Get, OnRequestXiaZai);
        request.Tag = pendingDownload;
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        if (pendingDownload.isRealtimeTrackingDownload)
        {
            realtimeTrackingModelDownloadRequests.Add(request);
        }
        LogHttpRequestStart("download-runtime-model", request);
        request.Send();
        ShowFrontMessage("download");
    }

    private void OnRequestXiaZai(HTTPRequest request, HTTPResponse response)
    {
        LogHttpRequestEnd("download-runtime-model", request, response);
        PendingModelDownload pendingDownload = request != null
            ? request.Tag as PendingModelDownload
            : null;
        if (request != null)
        {
            realtimeTrackingModelDownloadRequests.Remove(request);
        }
        if (pendingDownload != null && pendingDownload.cancelledByLocalHide)
        {
            ReleaseRuntimeModelDownload(pendingDownload);
            return;
        }
        if (response != null && response.IsSuccess)
        {
            if (pendingDownload == null || pendingDownload.instance == null || string.IsNullOrEmpty(pendingDownload.localPath))
            {
                Debug.LogError("[DOWNLOAD] Missing pending model download metadata.");
                ShowFrontMessage("download_ERR_missing_model_instance");
                ReleaseRuntimeModelDownload(pendingDownload);
                ResumeAsyncTaskQueueForDownload(pendingDownload);
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
                ReleaseRuntimeModelDownload(pendingDownload);
                ResumeAsyncTaskQueueForDownload(pendingDownload);
                return;
            }
            ShowFrontMessage("download " + receiver.Length);
            try
            {
                File.WriteAllBytes(pendingDownload.localPath, receiver);
            }
            catch (Exception exc)
            {
                Debug.LogError("[DOWNLOAD] Failed to cache runtime model: " + exc.Message);
                ShowFrontMessage("download_ERR_cache_write_failed");
                RuntimeModelManager cacheManager = RuntimeModelManager.Instance;
                if (cacheManager != null)
                {
                    cacheManager.DeleteCachedFile(pendingDownload.localPath);
                }
                ReleaseRuntimeModelDownload(pendingDownload);
                ResumeAsyncTaskQueueForDownload(pendingDownload);
                return;
            }
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
            ReleaseRuntimeModelDownload(pendingDownload);
            ResumeAsyncTaskQueueForDownload(pendingDownload);
        }
    }

    private void EnqueueRuntimeModelLoad(PendingModelDownload pendingDownload)
    {
        if (pendingDownload == null || pendingDownload.instance == null || string.IsNullOrEmpty(pendingDownload.localPath))
        {
            ReleaseRuntimeModelDownload(pendingDownload);
            ResumeAsyncTaskQueueForDownload(pendingDownload);
            return;
        }

        if (pendingDownload.isRealtimeTrackingDownload)
        {
            pendingModelLoadQueue.Enqueue(pendingDownload);
        }
        else
        {
            // HoloLens capture/ArUco-related model loads stay ahead of queued realtime
            // delivery work. The currently importing model is allowed to finish.
            int queuedCount = pendingModelLoadQueue.Count;
            bool inserted = false;
            for (int i = 0; i < queuedCount; i++)
            {
                PendingModelDownload queued = pendingModelLoadQueue.Dequeue();
                if (!inserted && queued != null && queued.isRealtimeTrackingDownload)
                {
                    pendingModelLoadQueue.Enqueue(pendingDownload);
                    inserted = true;
                }
                pendingModelLoadQueue.Enqueue(queued);
            }
            if (!inserted)
            {
                pendingModelLoadQueue.Enqueue(pendingDownload);
            }
        }
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
        else
        {
            RevealCurrentGenerationModelIfAllowed(completedLoad);
            ReleaseRuntimeModelDownload(completedLoad);
            ResumeAsyncTaskQueueForDownload(completedLoad);
        }

        ProcessNextQueuedRuntimeModelLoad();
    }

    private void RevealCurrentGenerationModelIfAllowed(PendingModelDownload completedLoad)
    {
        if (completedLoad == null
            || completedLoad.hideWhenComplete
            || completedLoad.displayGeneration != localModelDisplayGeneration
            || completedLoad.instance == null)
        {
            return;
        }
        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null)
        {
            return;
        }
        RuntimeModelRecord record;
        if (!string.IsNullOrEmpty(completedLoad.instance.ModelKey)
            && manager.TryGetLoadedRecord(completedLoad.instance.ModelKey, out record))
        {
            if (record != null && record.RootGameObject != null)
            {
                record.RootGameObject.SetActive(true);
            }
        }
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

        ReleaseRuntimeModelDownload(failedLoad);
        if (failedLoad.loadedFromCache
            && !failedLoad.hideWhenComplete
            && failedLoad.instance != null
            && !string.IsNullOrEmpty(failedLoad.instance.FbxUrl))
        {
            Debug.LogWarning("[DOWNLOAD] Cached model import failed; retry with HTTP: "
                + failedLoad.downloadIdentityKey);
            pendingModelInstance = failedLoad.instance;
            pendingModelShouldPlaceDebugMarkers = failedLoad.showDebugMarkers;
            DownloadPendingRuntimeModel(
                failedLoad.mayResumeAsyncTaskQueue,
                failedLoad.downloadIdentityKey);
            return;
        }
        ResumeAsyncTaskQueueForDownload(failedLoad);
    }

    private void ReleaseRuntimeModelDownload(PendingModelDownload pendingDownload)
    {
        if (pendingDownload == null)
        {
            return;
        }
        if (!string.IsNullOrEmpty(pendingDownload.downloadIdentityKey))
        {
            runtimeModelDownloadsInFlight.Remove(pendingDownload.downloadIdentityKey);
        }
        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager != null)
        {
            manager.UnprotectCachedModelPath(pendingDownload.localPath);
        }
    }

    private void ResumeAsyncTaskQueueForDownload(PendingModelDownload pendingDownload)
    {
        if (pendingDownload != null
            && pendingDownload.resumeAsyncTaskQueueWhenComplete
            && asyncTaskQueuePaused)
        {
            ResumeAsyncTaskQueuePolling();
        }
    }

}
