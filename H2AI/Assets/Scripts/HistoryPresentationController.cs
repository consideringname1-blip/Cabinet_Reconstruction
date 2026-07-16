using BestHTTP;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class HistoryPresentationController : MonoBehaviour
{
    [Tooltip(
        "Optional full history API override. Leave blank to derive it "
        + "from ShuJuQingQiu's realtime tracking service URL.")]
    [SerializeField] private string historyApiBaseUrl = "";

    private static HistoryPresentationController _instance;
    private readonly Dictionary<string, HTTPRequest> requestsByDisplayObjectId =
        new Dictionary<string, HTTPRequest>(StringComparer.Ordinal);
    private readonly Dictionary<string, long> requestGenerationByDisplayObjectId =
        new Dictionary<string, long>(StringComparer.Ordinal);
    private long globalGeneration;
    private bool globalReplayRequested;

    public static HistoryPresentationController Instance
    {
        get
        {
            if (_instance != null)
            {
                return _instance;
            }

            _instance = FindObjectOfType<HistoryPresentationController>();
            if (_instance == null)
            {
                GameObject controller =
                    new GameObject("HistoryPresentationController");
                _instance =
                    controller.AddComponent<HistoryPresentationController>();
            }
            return _instance;
        }
    }

    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(this);
            return;
        }
        _instance = this;
    }

    private void OnDestroy()
    {
        if (_instance == this)
        {
            _instance = null;
        }
        globalGeneration++;
        CancelAllRequests();
    }

    public void OnModelClicked(string displayObjectId)
    {
        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null
            || string.IsNullOrEmpty(displayObjectId)
            || !manager.TryGetLoadedRecordByDisplayObjectId(
                displayObjectId,
                out RuntimeModelRecord record)
            || record == null
            || record.RootGameObject == null)
        {
            ShowFrontMessage("history_presentation_model_missing");
            return;
        }

        string beforeCursor = "";
        if (manager.TryGetPresentationState(
                displayObjectId,
                out RuntimeObjectPresentationState state)
            && state != null)
        {
            beforeCursor = state.Mode == RuntimePresentationMode.History
                ? state.HistoryCursor
                : state.LatestLive.LatestOriginCursor;
        }
        RequestHistory(
            displayObjectId,
            beforeCursor,
            globalGeneration,
            false);
    }

    public void ToggleAllPresentations()
    {
        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager != null
            && (manager.IsAnyDisplayObjectInHistory()
                || globalReplayRequested))
        {
            ResumeAllPresentations(true);
            return;
        }
        ShowLatestPlacements();
    }

    public void ShowLatestPlacements()
    {
        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null)
        {
            ShowFrontMessage("history_presentation_manager_missing");
            return;
        }

        List<string> displayObjectIds = manager.GetDisplayObjectIds();
        if (displayObjectIds.Count == 0)
        {
            ShowFrontMessage("history_presentation_no_models");
            return;
        }

        globalGeneration++;
        CancelAllRequests();
        globalReplayRequested = true;
        long batchGeneration = globalGeneration;
        foreach (string displayObjectId in displayObjectIds)
        {
            RequestHistory(
                displayObjectId,
                "",
                batchGeneration,
                true);
        }
        FinishGlobalBatchIfNeeded();
        ShowFrontMessage(
            "history_presentation_loading_all_"
            + displayObjectIds.Count.ToString());
    }

    public int ResumeAllPresentations(bool showMessage)
    {
        globalGeneration++;
        globalReplayRequested = false;
        CancelAllRequests();
        RuntimeModelManager manager = RuntimeModelManager.Instance;
        int resumed = manager != null
            ? manager.ResumeAllLivePresentations()
            : 0;
        if (showMessage)
        {
            ShowFrontMessage(
                "history_presentation_live_all_" + resumed.ToString());
        }
        return resumed;
    }

    private void RequestHistory(
        string displayObjectId,
        string beforeCursor,
        long batchGeneration,
        bool wasGlobalRequest)
    {
        string startupSessionId = ShuJuQingQiu.initialize != null
            ? ShuJuQingQiu.initialize.startup_session_id
            : "";
        if (string.IsNullOrEmpty(startupSessionId))
        {
            ShowFrontMessage("history_presentation_session_missing");
            return;
        }

        long objectGeneration = NextObjectGeneration(displayObjectId);
        if (requestsByDisplayObjectId.TryGetValue(
                displayObjectId,
                out HTTPRequest existing)
            && existing != null)
        {
            existing.Abort();
        }

        if (!TryResolveHistoryApiBaseUrl(out string baseUrl))
        {
            ShowFrontMessage("history_presentation_url_invalid");
            return;
        }
        string url = baseUrl
            + "/display-objects/"
            + Uri.EscapeDataString(displayObjectId)
            + "/history?kind=origin&limit=1&startup_session_id="
            + Uri.EscapeDataString(startupSessionId);
        if (!string.IsNullOrEmpty(beforeCursor))
        {
            url += "&before_cursor=" + Uri.EscapeDataString(beforeCursor);
        }
        if (!Uri.TryCreate(url, UriKind.Absolute, out Uri requestUri))
        {
            ShowFrontMessage("history_presentation_url_invalid");
            return;
        }

        HistoryRequestContext context = new HistoryRequestContext
        {
            DisplayObjectId = displayObjectId,
            BeforeCursor = beforeCursor,
            StartupSessionId = startupSessionId,
            ObjectGeneration = objectGeneration,
            GlobalGeneration = batchGeneration,
            WasGlobalRequest = wasGlobalRequest,
        };
        HTTPRequest request = new HTTPRequest(
            requestUri,
            HTTPMethods.Get,
            OnHistoryRequestFinished);
        request.ConnectTimeout = TimeSpan.FromSeconds(2);
        request.Timeout = TimeSpan.FromSeconds(8);
        request.Tag = context;
        request.AddHeader("Accept", "application/json");
        requestsByDisplayObjectId[displayObjectId] = request;
        Debug.Log(
            "[HistoryPresentation] GET " + requestUri.AbsoluteUri);
        request.Send();
        if (!wasGlobalRequest)
        {
            ShowFrontMessage("history_presentation_loading");
        }
    }

    private bool TryResolveHistoryApiBaseUrl(out string baseUrl)
    {
        baseUrl = "";
        if (!string.IsNullOrWhiteSpace(historyApiBaseUrl))
        {
            string overrideUrl = historyApiBaseUrl.Trim().TrimEnd('/');
            if (!Uri.TryCreate(
                    overrideUrl,
                    UriKind.Absolute,
                    out Uri overrideUri)
                || (overrideUri.Scheme != Uri.UriSchemeHttp
                    && overrideUri.Scheme != Uri.UriSchemeHttps))
            {
                return false;
            }
            baseUrl = overrideUri.AbsoluteUri.TrimEnd('/');
            return true;
        }

        ShuJuQingQiu server = ShuJuQingQiu.initialize;
        if (server == null
            || !server.TryGetServerServiceBaseUri(out Uri serviceBaseUri))
        {
            return false;
        }

        Uri historyApiUri = new Uri(serviceBaseUri, "api/v2/");
        baseUrl = historyApiUri.AbsoluteUri.TrimEnd('/');
        return true;
    }

    private void OnHistoryRequestFinished(
        HTTPRequest request,
        HTTPResponse response)
    {
        HistoryRequestContext context = request != null
            ? request.Tag as HistoryRequestContext
            : null;
        if (context == null
            || !requestsByDisplayObjectId.TryGetValue(
                context.DisplayObjectId,
                out HTTPRequest current)
            || current != request)
        {
            return;
        }
        requestsByDisplayObjectId.Remove(context.DisplayObjectId);

        if (context.GlobalGeneration != globalGeneration
            || !requestGenerationByDisplayObjectId.TryGetValue(
                context.DisplayObjectId,
                out long currentObjectGeneration)
            || currentObjectGeneration != context.ObjectGeneration)
        {
            FinishGlobalBatchIfNeeded();
            return;
        }
        if (response == null || !response.IsSuccess)
        {
            Debug.LogWarning(
                "[HistoryPresentation] History request failed for "
                + context.DisplayObjectId
                + " response="
                + (response != null ? response.DataAsText : "null"));
            ShowFrontMessage("history_presentation_ERR_request");
            FinishGlobalBatchIfNeeded();
            return;
        }

        JObject root;
        try
        {
            root = JObject.Parse(response.DataAsText);
        }
        catch (Exception exception)
        {
            Debug.LogWarning(
                "[HistoryPresentation] Invalid JSON: " + exception.Message);
            ShowFrontMessage("history_presentation_ERR_invalid_response");
            FinishGlobalBatchIfNeeded();
            return;
        }

        if (!TryApplyHistoryResponse(context, root))
        {
            FinishGlobalBatchIfNeeded();
            return;
        }
        FinishGlobalBatchIfNeeded();
    }

    private bool TryApplyHistoryResponse(
        HistoryRequestContext context,
        JObject root)
    {
        JToken successToken = root != null ? root["success"] : null;
        string coordinateSpace = ReadString(root, "coordinate_space");
        string coordinateEpoch = ReadString(root, "coordinate_epoch");
        if (successToken == null
            || successToken.Type != JTokenType.Boolean
            || !successToken.Value<bool>()
            || coordinateSpace != "hololens_current_local"
            || string.IsNullOrEmpty(coordinateEpoch))
        {
            ShowFrontMessage("history_presentation_ERR_invalid_response");
            return false;
        }

        JToken eventToken = root["history_event"];
        if (eventToken == null || eventToken.Type == JTokenType.Null)
        {
            // Reaching the oldest cursor returns to the latest original/live
            // pose. Re-requesting the newest history row can return the row
            // already on screen and makes repeated clicks appear frozen.
            return TryApplyLatestLiveFallback(
                context,
                root,
                coordinateEpoch);
        }
        JObject historyEvent = eventToken as JObject;
        if (historyEvent == null)
        {
            ShowFrontMessage("history_presentation_ERR_invalid_response");
            return false;
        }

        string displayObjectId =
            ReadString(historyEvent, "display_object_id");
        string eventUid = ReadString(historyEvent, "event_uid");
        string historyCursor = ReadString(historyEvent, "history_cursor");
        string displayTimeJst = ReadString(historyEvent, "display_time_jst");
        JObject poseObject = historyEvent["pose"] as JObject;
        if (displayObjectId != context.DisplayObjectId
            || string.IsNullOrEmpty(eventUid)
            || string.IsNullOrEmpty(historyCursor)
            || string.IsNullOrEmpty(displayTimeJst)
            || (!string.IsNullOrEmpty(context.BeforeCursor)
                && historyCursor == context.BeforeCursor)
            || !TryParsePose(
                poseObject,
                out RuntimeModelPoseData historyPose))
        {
            ShowFrontMessage("history_presentation_ERR_invalid_response");
            return false;
        }

        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null
            || !manager.TryGetLoadedRecordByDisplayObjectId(
                displayObjectId,
                out RuntimeModelRecord loadedRecord)
            || loadedRecord == null
            || loadedRecord.RootGameObject == null)
        {
            ShowFrontMessage("history_presentation_model_missing");
            return false;
        }
        if (!manager.EnterHistoryPresentation(
                displayObjectId,
                eventUid,
                historyCursor,
                coordinateEpoch,
                historyPose,
                displayTimeJst,
                out string rejectionReason))
        {
            Debug.LogWarning(
                "[HistoryPresentation] Presentation rejected: "
                + rejectionReason);
            ShowFrontMessage("history_presentation_ERR_rejected");
            return false;
        }
        manager.UpdateDisplayObjectLatestOriginCursor(
            displayObjectId,
            ReadString(root, "latest_origin_cursor"));

        ShowFrontMessage("history_presentation_history");
        Debug.Log(
            "[HistoryPresentation] Applied history event="
            + eventUid
            + " display="
            + displayObjectId
            + " coordinate_epoch="
            + coordinateEpoch
            + " display_time_jst="
            + displayTimeJst
            + " position="
            + historyPose.HololensPosition.ToString("F4"));
        return true;
    }

    private bool TryApplyLatestLiveFallback(
        HistoryRequestContext context,
        JObject root,
        string coordinateEpoch)
    {
        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager == null)
        {
            ShowFrontMessage("history_presentation_manager_missing");
            return false;
        }

        if (!manager.AnimateToLatestLivePose(
                context.DisplayObjectId,
                coordinateEpoch,
                out string rejectionReason))
        {
            Debug.LogWarning(
                "[HistoryPresentation] Original-pose fallback rejected for "
                + context.DisplayObjectId
                + ": "
                + rejectionReason);
            ShowFrontMessage("history_presentation_ERR_rejected");
            return false;
        }
        manager.UpdateDisplayObjectLatestOriginCursor(
            context.DisplayObjectId,
            ReadString(root, "latest_origin_cursor"));

        ShowFrontMessage("history_presentation_original");
        Debug.Log(
            "[HistoryPresentation] No origin row; animated latest live pose "
            + "for display="
            + context.DisplayObjectId
            + " coordinate_epoch="
            + coordinateEpoch);
        return true;
    }

    private static bool TryParsePose(
        JObject pose,
        out RuntimeModelPoseData parsed)
    {
        parsed = null;
        if (pose == null
            || !HasExactKeys(
                pose,
                "position",
                "rotation_quaternion_xyzw")
            || !TryReadVector3(pose["position"], out Vector3 position)
            || !TryReadQuaternion(
                pose["rotation_quaternion_xyzw"],
                out Quaternion rotation))
        {
            return false;
        }
        parsed = new RuntimeModelPoseData
        {
            HasHololensPose = true,
            HololensPosition = position,
            HololensRotation = rotation,
        };
        return true;
    }

    private static bool TryParseSpatialBox(
        JObject box,
        string coordinateEpoch,
        out RuntimeSpatialBoxData parsed)
    {
        parsed = null;
        if (box == null
            || !HasExactKeys(
                box,
                "status",
                "coordinate_space",
                "revision",
                "corners_hololens_current_local_m")
            || ReadString(box, "status") != "ready"
            || ReadString(box, "coordinate_space")
                != "hololens_current_local")
        {
            return false;
        }
        JToken revisionToken = box["revision"];
        JArray corners =
            box["corners_hololens_current_local_m"] as JArray;
        if (revisionToken == null
            || revisionToken.Type != JTokenType.Integer
            || revisionToken.Value<long>() <= 0
            || corners == null
            || corners.Count != 8)
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
        parsed = new RuntimeSpatialBoxData
        {
            IsReady = true,
            Status = "ready",
            CoordinateSpace = "hololens_current_local",
            CoordinateEpoch = coordinateEpoch,
            Revision = revisionToken.Value<long>(),
            CornersWorld = parsedCorners,
        };
        return true;
    }

    private static bool TryReadVector3(
        JToken token,
        out Vector3 value)
    {
        value = Vector3.zero;
        JArray array = token as JArray;
        if (array == null
            || array.Count != 3
            || !IsJsonNumber(array[0])
            || !IsJsonNumber(array[1])
            || !IsJsonNumber(array[2]))
        {
            return false;
        }
        value = new Vector3(
            array[0].Value<float>(),
            array[1].Value<float>(),
            array[2].Value<float>());
        return IsFinite(value);
    }

    private static bool TryReadQuaternion(
        JToken token,
        out Quaternion value)
    {
        value = Quaternion.identity;
        JArray array = token as JArray;
        if (array == null
            || array.Count != 4
            || !IsJsonNumber(array[0])
            || !IsJsonNumber(array[1])
            || !IsJsonNumber(array[2])
            || !IsJsonNumber(array[3]))
        {
            return false;
        }
        float x = array[0].Value<float>();
        float y = array[1].Value<float>();
        float z = array[2].Value<float>();
        float w = array[3].Value<float>();
        float norm = Mathf.Sqrt(x * x + y * y + z * z + w * w);
        if (float.IsNaN(norm)
            || float.IsInfinity(norm)
            || norm <= 0.000001f)
        {
            return false;
        }
        value = new Quaternion(x / norm, y / norm, z / norm, w / norm);
        return true;
    }

    private static bool IsJsonNumber(JToken token)
    {
        return token != null
            && (token.Type == JTokenType.Integer
                || token.Type == JTokenType.Float);
    }

    private static bool IsFinite(Vector3 value)
    {
        return !float.IsNaN(value.x)
            && !float.IsInfinity(value.x)
            && !float.IsNaN(value.y)
            && !float.IsInfinity(value.y)
            && !float.IsNaN(value.z)
            && !float.IsInfinity(value.z);
    }

    private static string ReadString(JObject payload, string key)
    {
        JToken token = payload != null ? payload[key] : null;
        return token != null && token.Type == JTokenType.String
            ? token.Value<string>().Trim()
            : "";
    }

    private static bool HasExactKeys(
        JObject payload,
        params string[] expectedKeys)
    {
        if (payload == null || payload.Count != expectedKeys.Length)
        {
            return false;
        }
        HashSet<string> expected =
            new HashSet<string>(expectedKeys, StringComparer.Ordinal);
        foreach (JProperty property in payload.Properties())
        {
            if (!expected.Remove(property.Name))
            {
                return false;
            }
        }
        return expected.Count == 0;
    }

    private long NextObjectGeneration(string displayObjectId)
    {
        requestGenerationByDisplayObjectId.TryGetValue(
            displayObjectId,
            out long current);
        current++;
        requestGenerationByDisplayObjectId[displayObjectId] = current;
        return current;
    }

    private void CancelAllRequests()
    {
        foreach (HTTPRequest request in
            new List<HTTPRequest>(requestsByDisplayObjectId.Values))
        {
            if (request != null)
            {
                request.Abort();
            }
        }
        requestsByDisplayObjectId.Clear();
    }

    private void FinishGlobalBatchIfNeeded()
    {
        if (globalReplayRequested && requestsByDisplayObjectId.Count == 0)
        {
            globalReplayRequested = false;
        }
    }

    private static void ShowFrontMessage(string message)
    {
        if (Game_M.initialize != null)
        {
            Game_M.initialize.XianShiForSeconds(message);
        }
    }

    private sealed class HistoryRequestContext
    {
        public string DisplayObjectId = "";
        public string BeforeCursor = "";
        public string StartupSessionId = "";
        public long ObjectGeneration;
        public long GlobalGeneration;
        public bool WasGlobalRequest;
    }
}
