using BestHTTP;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Globalization;
using System.Text;
using UnityEngine;

[DisallowMultipleComponent]
public class SpatialHistoryPointerQuery : MonoBehaviour
{
    public enum QueryHand
    {
        Left,
        Right,
    }

    [Header("Pointer")]
    [SerializeField] private QueryHand targetHand = QueryHand.Right;
    [SerializeField, Min(0f)] private float waitSeconds = 3f;
    [SerializeField, Min(0.1f)] private float maxDistanceMeters = 10f;
    [SerializeField] private LayerMask spatialRaycastMask = ~0;
    [SerializeField] private bool showRayLine = true;

    [Header("Server")]
    [SerializeField] private string serverBaseUrl = "http://10.40.1.122:7355";
    [SerializeField, Range(1, 50)] private int historyLimit = 5;

    [Header("References")]
    [SerializeField] private ShuJuQingQiu shuJuQingQiu;
    [SerializeField] private RuntimeModelManager runtimeModelManager;
    [SerializeField] private LoadModel loadModel;

    [Header("Debug Bounds")]
    [SerializeField] private Material lineMaterial;
    [SerializeField, Min(0.001f)] private float lineWidth = 0.008f;
    [SerializeField] private int debugBoundsLayer = 2;
    [SerializeField] private bool showBoundsLabels = true;
    [SerializeField] private Color rayValidColor = new Color(0.1f, 0.9f, 1.0f, 1f);
    [SerializeField] private Color rayInvalidColor = new Color(1f, 0.15f, 0.1f, 1f);
    [SerializeField]
    private Color[] debugBoundsColors =
    {
        new Color(0.1f, 0.9f, 1.0f, 1f),
        new Color(1.0f, 0.75f, 0.15f, 1f),
        new Color(0.3f, 1.0f, 0.35f, 1f),
        new Color(1.0f, 0.35f, 0.75f, 1f),
        new Color(0.7f, 0.5f, 1.0f, 1f),
    };

    private const string BoundsRootName = "SpatialHistoryBoundsDebugRoot";
    private LineRenderer rayLineRenderer;
    private GameObject boundsDebugRoot;
    private Coroutine queryCoroutine;
    private Material runtimeLineMaterial;

    private static readonly int[,] EdgePairs =
    {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
        { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
        { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
    };

    private void Awake()
    {
        ResolveReferences();
        EnsureLineMaterial();
        if (showRayLine)
        {
            EnsureRayLine();
        }
    }

    private void Start()
    {
        ResolveReferences();
    }

    private void Update()
    {
        if (showRayLine)
        {
            UpdateRayLine();
        }
        else if (rayLineRenderer != null)
        {
            rayLineRenderer.gameObject.SetActive(false);
        }
    }

    private void OnDestroy()
    {
        ClearBoundsDebug();
        if (rayLineRenderer != null)
        {
            Destroy(rayLineRenderer.gameObject);
            rayLineRenderer = null;
        }
    }

    private void ResolveReferences()
    {
        if (shuJuQingQiu == null)
        {
            shuJuQingQiu = ShuJuQingQiu.initialize != null
                ? ShuJuQingQiu.initialize
                : FindObjectOfType<ShuJuQingQiu>();
        }
        if (runtimeModelManager == null)
        {
            runtimeModelManager = RuntimeModelManager.Instance;
        }
        if (loadModel == null)
        {
            loadModel = FindObjectOfType<LoadModel>();
        }
    }

    private void EnsureLineMaterial()
    {
        if (lineMaterial != null)
        {
            runtimeLineMaterial = lineMaterial;
            return;
        }
        if (runtimeLineMaterial != null)
        {
            return;
        }

        Shader shader = Shader.Find("Sprites/Default");
        runtimeLineMaterial = shader != null ? new Material(shader) : new Material(Shader.Find("Standard"));
    }

    private Handedness TargetHandedness()
    {
        return targetHand == QueryHand.Right ? Handedness.Right : Handedness.Left;
    }

    private bool TryGetFingerRay(out Vector3 originWorld, out Vector3 directionWorld)
    {
        originWorld = Vector3.zero;
        directionWorld = Vector3.forward;

        Handedness handedness = TargetHandedness();
        if (!HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexTip, handedness, out MixedRealityPose tipPose)
            || !HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexDistalJoint, handedness, out MixedRealityPose distalPose))
        {
            return false;
        }

        Vector3 direction = tipPose.Position - distalPose.Position;
        if (direction.sqrMagnitude < 0.000001f)
        {
            direction = tipPose.Rotation * Vector3.forward;
        }
        if (direction.sqrMagnitude < 0.000001f)
        {
            return false;
        }

        originWorld = tipPose.Position;
        directionWorld = direction.normalized;
        return true;
    }

    private bool TryGetCurrentArucoReference(out Vector3 arucoPosition, out Quaternion arucoRotation)
    {
        ResolveReferences();
        arucoPosition = Vector3.zero;
        arucoRotation = Quaternion.identity;

        if (runtimeModelManager != null
            && runtimeModelManager.TryGetCurrentArucoReference(out arucoPosition, out arucoRotation))
        {
            return true;
        }

        if (shuJuQingQiu != null && shuJuQingQiu.hasArucoReferencePose)
        {
            arucoPosition = shuJuQingQiu.arucoReferencePosition;
            arucoRotation = shuJuQingQiu.arucoReferenceRotation;
            return true;
        }

        return false;
    }

    private Vector3 WorldPointToAruco(Vector3 worldPoint, Vector3 arucoPosition, Quaternion arucoRotation)
    {
        return Quaternion.Inverse(arucoRotation) * (worldPoint - arucoPosition);
    }

    private Vector3 WorldDirectionToAruco(Vector3 worldDirection, Quaternion arucoRotation)
    {
        return (Quaternion.Inverse(arucoRotation) * worldDirection).normalized;
    }

    private Vector3 ArucoPointToWorld(Vector3 arucoPoint, Vector3 arucoPosition, Quaternion arucoRotation)
    {
        return arucoPosition + (arucoRotation * arucoPoint);
    }

    private int RaycastMaskWithoutDebugBounds()
    {
        int mask = spatialRaycastMask.value;
        if (debugBoundsLayer >= 0 && debugBoundsLayer <= 31)
        {
            mask &= ~(1 << debugBoundsLayer);
        }
        return mask;
    }

    private Vector3 ResolveEndpoint(Vector3 originWorld, Vector3 directionWorld)
    {
        float distance = Mathf.Max(0.1f, maxDistanceMeters);
        int mask = RaycastMaskWithoutDebugBounds();
        if (Physics.Raycast(
            originWorld,
            directionWorld,
            out RaycastHit hit,
            distance,
            mask,
            QueryTriggerInteraction.Ignore))
        {
            return hit.point;
        }

        return originWorld + (directionWorld * distance);
    }

    private void EnsureRayLine()
    {
        if (rayLineRenderer != null)
        {
            return;
        }

        GameObject rayObject = new GameObject("SpatialHistoryPointerRay");
        rayObject.transform.SetParent(transform, false);
        ApplyDebugLayer(rayObject);
        rayLineRenderer = rayObject.AddComponent<LineRenderer>();
        rayLineRenderer.useWorldSpace = true;
        rayLineRenderer.positionCount = 2;
        rayLineRenderer.material = runtimeLineMaterial;
        rayLineRenderer.startWidth = lineWidth;
        rayLineRenderer.endWidth = lineWidth;
        rayLineRenderer.numCapVertices = 4;
    }

    private void UpdateRayLine()
    {
        EnsureLineMaterial();
        EnsureRayLine();

        if (!TryGetFingerRay(out Vector3 originWorld, out Vector3 directionWorld))
        {
            rayLineRenderer.startColor = rayInvalidColor;
            rayLineRenderer.endColor = rayInvalidColor;
            rayLineRenderer.gameObject.SetActive(false);
            return;
        }

        Vector3 endWorld = ResolveEndpoint(originWorld, directionWorld);
        rayLineRenderer.gameObject.SetActive(true);
        rayLineRenderer.SetPosition(0, originWorld);
        rayLineRenderer.SetPosition(1, endWorld);
        rayLineRenderer.startColor = rayValidColor;
        rayLineRenderer.endColor = rayValidColor;
    }

    public void TriggerSpatialHistoryQuery()
    {
        if (queryCoroutine != null)
        {
            StopCoroutine(queryCoroutine);
        }
        queryCoroutine = StartCoroutine(SpatialHistoryQueryCoroutine());
    }

    private IEnumerator SpatialHistoryQueryCoroutine()
    {
        if (waitSeconds > 0f)
        {
            yield return new WaitForSeconds(waitSeconds);
        }

        queryCoroutine = null;

        if (!TryGetFingerRay(out Vector3 originWorld, out Vector3 directionWorld))
        {
            Debug.LogWarning("[SpatialHistoryPointerQuery] Target hand is not tracked.");
            ShowFrontMessage("spatial_query_ERR_hand_not_tracked");
            yield break;
        }

        if (!TryGetCurrentArucoReference(out Vector3 arucoPosition, out Quaternion arucoRotation))
        {
            Debug.LogWarning("[SpatialHistoryPointerQuery] ArUco reference is not available.");
            ShowFrontMessage("spatial_query_ERR_no_aruco_reference");
            yield break;
        }

        Vector3 endWorld = ResolveEndpoint(originWorld, directionWorld);
        Vector3 originAruco = WorldPointToAruco(originWorld, arucoPosition, arucoRotation);
        Vector3 directionAruco = WorldDirectionToAruco(directionWorld, arucoRotation);
        Vector3 endAruco = WorldPointToAruco(endWorld, arucoPosition, arucoRotation);

        JObject payload = new JObject
        {
            ["origin_aruco"] = VectorToJArray(originAruco),
            ["direction_aruco"] = VectorToJArray(directionAruco),
            ["end_aruco"] = VectorToJArray(endAruco),
            ["max_distance_m"] = Mathf.Max(0.1f, maxDistanceMeters),
            ["limit"] = Mathf.Clamp(historyLimit, 1, 50),
        };
        SendSpatialRayRequest(payload);
    }

    private void SendSpatialRayRequest(JObject payload)
    {
        string url = NormalizeServerBaseUrl() + "/spatial-query/ray";
        HTTPRequest request = new HTTPRequest(new Uri(url), HTTPMethods.Post, OnSpatialRayResponse);
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.RawData = Encoding.UTF8.GetBytes(payload.ToString(Formatting.None));
        request.Send();
        ShowFrontMessage("spatial_query");
    }

    private void OnSpatialRayResponse(HTTPRequest request, HTTPResponse response)
    {
        if (response == null || !response.IsSuccess)
        {
            string statusCode = response != null ? response.StatusCode.ToString(CultureInfo.InvariantCulture) : "no_response";
            string message = response != null ? response.Message : "No response from server";
            Debug.LogError("[SpatialHistoryPointerQuery] Ray query failed: " + statusCode + " - " + message);
            ShowFrontMessage("spatial_query_ERR_request_failed");
            return;
        }

        JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        bool success = jo["success"] != null && jo["success"].Value<bool>();
        bool hit = jo["hit"] != null && jo["hit"].Value<bool>();
        if (!success)
        {
            Debug.LogWarning("[SpatialHistoryPointerQuery] Ray query returned success=false: " + response.DataAsText);
            ShowFrontMessage("spatial_query_ERR_server");
            return;
        }
        if (!hit)
        {
            Debug.Log("[SpatialHistoryPointerQuery] No model bounds were hit.");
            ShowFrontMessage("spatial_query_no_hit");
            return;
        }

        JObject modelJ = jo["model"] as JObject;
        if (modelJ == null)
        {
            ShowFrontMessage("spatial_query_ERR_missing_model");
            return;
        }

        ResolveReferences();
        if (shuJuQingQiu == null)
        {
            Debug.LogError("[SpatialHistoryPointerQuery] ShuJuQingQiu was not found.");
            ShowFrontMessage("spatial_query_ERR_no_loader");
            return;
        }

        shuJuQingQiu.DownloadRuntimeModelFromSpatialQueryModel(modelJ);
    }

    public void ToggleBoundsDebug()
    {
        if (boundsDebugRoot != null)
        {
            ClearBoundsDebug();
            return;
        }

        if (!TryGetCurrentArucoReference(out Vector3 arucoPosition, out Quaternion arucoRotation))
        {
            Debug.LogWarning("[SpatialHistoryPointerQuery] ArUco reference is not available; bounds debug skipped.");
            ShowFrontMessage("bounds_debug_ERR_no_aruco_reference");
            return;
        }

        string url = NormalizeServerBaseUrl()
            + "/model-bounds/latest?limit="
            + Mathf.Clamp(historyLimit, 1, 50).ToString(CultureInfo.InvariantCulture);
        HTTPRequest request = new HTTPRequest(new Uri(url), HTTPMethods.Get, OnBoundsLatestResponse);
        request.Tag = new Tuple<Vector3, Quaternion>(arucoPosition, arucoRotation);
        request.AddHeader("Content-Type", "application/json;charset=UTF-8");
        request.Send();
        ShowFrontMessage("bounds_debug_loading");
    }

    private void OnBoundsLatestResponse(HTTPRequest request, HTTPResponse response)
    {
        if (response == null || !response.IsSuccess)
        {
            string statusCode = response != null ? response.StatusCode.ToString(CultureInfo.InvariantCulture) : "no_response";
            string message = response != null ? response.Message : "No response from server";
            Debug.LogError("[SpatialHistoryPointerQuery] Bounds latest failed: " + statusCode + " - " + message);
            ShowFrontMessage("bounds_debug_ERR_request_failed");
            return;
        }

        Tuple<Vector3, Quaternion> reference = request.Tag as Tuple<Vector3, Quaternion>;
        Vector3 arucoPosition = reference != null ? reference.Item1 : Vector3.zero;
        Quaternion arucoRotation = reference != null ? reference.Item2 : Quaternion.identity;

        JObject jo = (JObject)JsonConvert.DeserializeObject(response.DataAsText);
        JArray boundsArray = jo["bounds"] as JArray;
        if (boundsArray == null)
        {
            ShowFrontMessage("bounds_debug_ERR_bad_response");
            return;
        }

        ClearBoundsDebug();
        boundsDebugRoot = new GameObject(BoundsRootName);
        ApplyDebugLayer(boundsDebugRoot);

        for (int i = 0; i < boundsArray.Count; i++)
        {
            JObject boundJ = boundsArray[i] as JObject;
            if (boundJ == null)
            {
                continue;
            }
            if (!TryReadCorners(boundJ["corners_aruco"], out Vector3[] cornersAruco))
            {
                continue;
            }

            Color color = debugBoundsColors != null && debugBoundsColors.Length > 0
                ? debugBoundsColors[i % debugBoundsColors.Length]
                : Color.cyan;
            Vector3[] cornersWorld = new Vector3[cornersAruco.Length];
            for (int c = 0; c < cornersAruco.Length; c++)
            {
                cornersWorld[c] = ArucoPointToWorld(cornersAruco[c], arucoPosition, arucoRotation);
            }

            CreateBoundsWireframe(boundJ, cornersWorld, color, i);
        }

        ShowFrontMessage("bounds_debug_on");
    }

    private void CreateBoundsWireframe(JObject boundJ, Vector3[] cornersWorld, Color color, int index)
    {
        GameObject itemRoot = new GameObject("SpatialHistoryBounds_" + index.ToString(CultureInfo.InvariantCulture));
        itemRoot.transform.SetParent(boundsDebugRoot.transform, false);
        ApplyDebugLayer(itemRoot);

        for (int e = 0; e < EdgePairs.GetLength(0); e++)
        {
            int startIndex = EdgePairs[e, 0];
            int endIndex = EdgePairs[e, 1];
            CreateEdgeLine(itemRoot.transform, cornersWorld[startIndex], cornersWorld[endIndex], color, e);
        }

        if (showBoundsLabels)
        {
            CreateBoundsLabel(itemRoot.transform, boundJ, cornersWorld, color);
        }
    }

    private void CreateEdgeLine(Transform parent, Vector3 start, Vector3 end, Color color, int edgeIndex)
    {
        GameObject edgeObject = new GameObject("edge_" + edgeIndex.ToString(CultureInfo.InvariantCulture));
        edgeObject.transform.SetParent(parent, false);
        ApplyDebugLayer(edgeObject);
        LineRenderer line = edgeObject.AddComponent<LineRenderer>();
        line.useWorldSpace = true;
        line.positionCount = 2;
        line.material = runtimeLineMaterial;
        line.startWidth = lineWidth;
        line.endWidth = lineWidth;
        line.startColor = color;
        line.endColor = color;
        line.numCapVertices = 2;
        line.SetPosition(0, start);
        line.SetPosition(1, end);
    }

    private void CreateBoundsLabel(Transform parent, JObject boundJ, Vector3[] cornersWorld, Color color)
    {
        Vector3 center = Vector3.zero;
        for (int i = 0; i < cornersWorld.Length; i++)
        {
            center += cornersWorld[i];
        }
        center /= Mathf.Max(1, cornersWorld.Length);

        string label = boundJ["model_name"]?.ToString();
        if (string.IsNullOrEmpty(label))
        {
            label = boundJ["task_id"]?.ToString();
        }
        if (string.IsNullOrEmpty(label))
        {
            label = "model";
        }
        if (label.Length > 18)
        {
            label = label.Substring(0, 18);
        }

        GameObject labelObject = new GameObject("label");
        labelObject.transform.SetParent(parent, false);
        labelObject.transform.position = center + Vector3.up * 0.08f;
        ApplyDebugLayer(labelObject);
        TextMesh text = labelObject.AddComponent<TextMesh>();
        text.text = label;
        text.anchor = TextAnchor.MiddleCenter;
        text.alignment = TextAlignment.Center;
        text.characterSize = 0.04f;
        text.fontSize = 48;
        text.color = color;

        Camera cam = Camera.main;
        if (cam != null)
        {
            labelObject.transform.rotation = Quaternion.LookRotation(
                labelObject.transform.position - cam.transform.position,
                Vector3.up
            );
        }
    }

    private bool TryReadCorners(JToken token, out Vector3[] corners)
    {
        corners = null;
        JArray arr = token as JArray;
        if (arr == null || arr.Count != 8)
        {
            return false;
        }

        Vector3[] parsed = new Vector3[8];
        for (int i = 0; i < arr.Count; i++)
        {
            if (!TryReadVector3(arr[i], out parsed[i]))
            {
                return false;
            }
        }

        corners = parsed;
        return true;
    }

    private bool TryReadVector3(JToken token, out Vector3 value)
    {
        value = Vector3.zero;
        JArray arr = token as JArray;
        if (arr == null || arr.Count < 3)
        {
            return false;
        }

        value = new Vector3(arr[0].Value<float>(), arr[1].Value<float>(), arr[2].Value<float>());
        return true;
    }

    private JArray VectorToJArray(Vector3 value)
    {
        return new JArray(value.x, value.y, value.z);
    }

    private string NormalizeServerBaseUrl()
    {
        string value = string.IsNullOrEmpty(serverBaseUrl) ? "http://10.40.1.122:7355" : serverBaseUrl.Trim();
        return value.EndsWith("/", StringComparison.Ordinal) ? value.Substring(0, value.Length - 1) : value;
    }

    private void ApplyDebugLayer(GameObject target)
    {
        if (target != null && debugBoundsLayer >= 0 && debugBoundsLayer <= 31)
        {
            target.layer = debugBoundsLayer;
        }
    }

    private void ClearBoundsDebug()
    {
        if (boundsDebugRoot != null)
        {
            Destroy(boundsDebugRoot);
            boundsDebugRoot = null;
            ShowFrontMessage("bounds_debug_off");
        }
    }

    private void ShowFrontMessage(string message)
    {
        if (Game_M.initialize != null)
        {
            Game_M.initialize.XianShi(message);
        }
    }
}
