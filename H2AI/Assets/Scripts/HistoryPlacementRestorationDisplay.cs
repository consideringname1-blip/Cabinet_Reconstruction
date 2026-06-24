using BestHTTP;
using Microsoft.MixedReality.Toolkit.Input;
using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class HistoryPlacementRestorationDisplay : MonoBehaviour
{
    private const string RootName = "HistoryPlacementRestorationDisplayRoot";
    private const float PolyhedronTopClearanceMeters = 0.30f;
    private const float EvidenceImageVerticalOffsetMeters = 0.30f;
    private const float EvidenceImageSideOffsetMeters = 0.24f;
    private const float EvidenceImageLerpSpeed = 8.0f;

    private static HistoryPlacementRestorationDisplay _instance;
    private readonly Dictionary<string, HistoryPlacementRestorationItem> activeItems = new Dictionary<string, HistoryPlacementRestorationItem>();
    private GameObject activeRoot;
    private Material cubeMaterial;
    private Material tetrahedronMaterial;
    private Material octahedronMaterial;
    private Material dodecahedronMaterial;
    private Material icosahedronMaterial;
    private Material evidenceBodyMaterial;
    private RuntimeModelManager runtimeModelManager;

    public static HistoryPlacementRestorationDisplay Instance
    {
        get
        {
            if (_instance != null)
            {
                return _instance;
            }

            _instance = FindObjectOfType<HistoryPlacementRestorationDisplay>();
            if (_instance == null)
            {
                GameObject displayObject = new GameObject("HistoryPlacementRestorationDisplay");
                _instance = displayObject.AddComponent<HistoryPlacementRestorationDisplay>();
            }
            return _instance;
        }
    }

    public bool HasActiveDisplay
    {
        get { return activeRoot != null; }
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

    private void Update()
    {
        foreach (HistoryPlacementRestorationItem item in activeItems.Values)
        {
            if (item != null && item.EvidenceImageObject != null && item.EvidenceImageObject.activeSelf)
            {
                UpdateEvidenceImagePlacement(item, false);
            }
        }
    }

    private void OnDestroy()
    {
        if (_instance == this)
        {
            _instance = null;
        }
        Clear();
        DestroyMaterial(evidenceBodyMaterial);
        evidenceBodyMaterial = null;
    }

    public int ShowFromServerResponse(JObject response)
    {
        Clear();
        ResolveRuntimeModelManager();
        if (runtimeModelManager == null)
        {
            ShowFrontMessage("history_placement_restoration_ERR_no_runtime_manager");
            return 0;
        }

        if (!TryGetArucoReference(out Vector3 arucoPosition, out Quaternion arucoRotation))
        {
            ShowFrontMessage("history_placement_restoration_ERR_no_aruco_reference");
            return 0;
        }

        JArray results = response != null ? response["results"] as JArray : null;
        if (results == null || results.Count == 0)
        {
            ShowFrontMessage("history_placement_restoration_no_result");
            return 0;
        }

        activeRoot = new GameObject(RootName);
        int createdCount = 0;
        foreach (JToken token in results)
        {
            JObject result = token as JObject;
            if (result == null || result["success"] == null || !result["success"].Value<bool>())
            {
                continue;
            }

            JObject payload = result["history_placement_restoration"] as JObject;
            if (payload == null)
            {
                continue;
            }

            string taskId = result["task_id"] != null ? result["task_id"].ToString() : "";
            if (string.IsNullOrEmpty(taskId))
            {
                taskId = payload["task_id"] != null ? payload["task_id"].ToString() : "";
            }

            if (CreateDisplayForPayload(taskId, result, payload, arucoPosition, arucoRotation))
            {
                createdCount++;
            }
        }

        if (createdCount <= 0)
        {
            Clear();
        }
        return createdCount;
    }

    public void Clear()
    {
        foreach (HistoryPlacementRestorationItem item in new List<HistoryPlacementRestorationItem>(activeItems.Values))
        {
            if (item != null && item.AnimationCoroutine != null)
            {
                StopCoroutine(item.AnimationCoroutine);
            }
            if (item != null && item.BodyVisibilityCoroutine != null)
            {
                StopCoroutine(item.BodyVisibilityCoroutine);
            }
            if (item != null && item.EvidenceTexture != null)
            {
                Destroy(item.EvidenceTexture);
                item.EvidenceTexture = null;
            }
        }
        activeItems.Clear();
        if (activeRoot != null)
        {
            Destroy(activeRoot);
            activeRoot = null;
        }
    }

    public void HandlePolyhedronClicked(string itemKey)
    {
        if (string.IsNullOrEmpty(itemKey) || !activeItems.ContainsKey(itemKey))
        {
            return;
        }

        HistoryPlacementRestorationItem item = activeItems[itemKey];
        if (item == null)
        {
            return;
        }

        bool evidenceHandled = EnsureEvidenceForItem(item);
        if (!item.HasAnimation)
        {
            if (!evidenceHandled)
            {
                ShowFrontMessage("history_placement_restoration_no_animation_model");
            }
            return;
        }

        if (item.CloneObject != null && item.CloneObject.activeSelf)
        {
            if (item.AnimationCoroutine != null)
            {
                StopCoroutine(item.AnimationCoroutine);
                item.AnimationCoroutine = null;
            }
            item.CloneObject.SetActive(false);
            return;
        }

        if (item.CloneObject == null)
        {
            item.CloneObject = CreateAnimationClone(item.TaskId, item.FromPosition, item.FromRotation);
        }

        if (item.CloneObject == null)
        {
            if (QueueModelDownloadForItem(item))
            {
                if (item.AnimationCoroutine != null)
                {
                    StopCoroutine(item.AnimationCoroutine);
                }
                item.AnimationCoroutine = StartCoroutine(WaitForModelThenAnimate(item));
                ShowFrontMessage("history_placement_restoration_downloading_model");
                return;
            }
            ShowFrontMessage("history_placement_restoration_no_animation_model");
            return;
        }

        StartItemAnimation(item);
    }

    private void StartItemAnimation(HistoryPlacementRestorationItem item)
    {
        if (item == null || item.CloneObject == null)
        {
            return;
        }
        if (item.AnimationCoroutine != null)
        {
            StopCoroutine(item.AnimationCoroutine);
        }
        item.CloneObject.SetActive(true);
        item.CloneObject.transform.SetPositionAndRotation(item.FromPosition, item.FromRotation);
        item.AnimationCoroutine = StartCoroutine(AnimateCloneToOriginal(item));
    }

    public bool ToggleEvidenceForModel(string taskIdOrModelKey)
    {
        HistoryPlacementRestorationItem item = FindEvidenceItem(taskIdOrModelKey);
        if (item == null)
        {
            return false;
        }

        if (IsEvidenceVisibleOrPending(item))
        {
            HideEvidenceForItem(item);
            return true;
        }
        if (EnsureEvidenceForItem(item))
        {
            return true;
        }
        ShowFrontMessage("history_placement_restoration_no_evidence");
        return true;
    }

    public bool RegisterEvidenceForModel(JObject result)
    {
        if (result == null)
        {
            return false;
        }

        JObject modelInstance = result["model_instance"] as JObject;
        string taskId = result["task_id"]?.ToString() ?? modelInstance?["task_id"]?.ToString() ?? "";
        string modelKey = modelInstance?["model_key"]?.ToString() ?? "";
        string itemKey = !string.IsNullOrEmpty(taskId) ? taskId : modelKey;
        if (string.IsNullOrEmpty(itemKey))
        {
            return false;
        }

        HistoryPlacementRestorationItem item;
        if (!activeItems.TryGetValue(itemKey, out item) || item == null)
        {
            item = new HistoryPlacementRestorationItem
            {
                Key = itemKey,
                DurationSeconds = 1.2f,
            };
        }

        item.TaskId = taskId;
        item.ModelKey = modelKey;
        item.Status = result["status"]?.ToString() ?? item.Status;
        item.DownloadModel = BuildDownloadModel(taskId, result);
        item.TakenRgbUrl = ReadNestedString(result, "taken_object_detection_urls", "result_rgb_url");
        item.BodyFbxUrl = ReadNestedString(result, "sam3d_body_mesh_urls", "selected_person_fbx_url");
        item.BodyModelKey = "body:" + itemKey;
        item.ArucoReference = CloneOrNull(result["aruco_reference"]);
        item.ObjectAruco = CloneOrNull(result["object_aruco"]);
        JObject bodyPayload = result["sam3d_body_mesh"] as JObject;
        item.HasBodyObjectCenterAruco = bodyPayload != null && TryReadVector3(bodyPayload["object_center_armarker"], out item.BodyObjectCenterAruco);

        ResolveRuntimeModelManager();
        if (TryGetArucoReference(out Vector3 arucoPosition, out Quaternion arucoRotation)
            && TryReadFallbackPolyhedronPose(result, arucoPosition, arucoRotation, out Vector3 anchorPosition, out Quaternion anchorRotation))
        {
            item.ToPosition = anchorPosition;
            item.ToRotation = anchorRotation;
        }

        activeItems[itemKey] = item;
        return true;
    }

    private HistoryPlacementRestorationItem FindEvidenceItem(string taskIdOrModelKey)
    {
        if (string.IsNullOrEmpty(taskIdOrModelKey))
        {
            return null;
        }

        HistoryPlacementRestorationItem item;
        if (activeItems.TryGetValue(taskIdOrModelKey, out item) && item != null)
        {
            return item;
        }

        foreach (HistoryPlacementRestorationItem candidate in activeItems.Values)
        {
            if (candidate == null)
            {
                continue;
            }
            if (candidate.TaskId == taskIdOrModelKey
                || candidate.Key == taskIdOrModelKey
                || candidate.ModelKey == taskIdOrModelKey
                || candidate.BodyModelKey == taskIdOrModelKey)
            {
                return candidate;
            }
        }
        return null;
    }

    private bool IsEvidenceVisibleOrPending(HistoryPlacementRestorationItem item)
    {
        if (item == null)
        {
            return false;
        }
        if (item.ImageRequestInFlight || item.BodyDownloadQueued)
        {
            return true;
        }
        if (item.EvidenceImageObject != null && item.EvidenceImageObject.activeSelf)
        {
            return true;
        }

        ResolveRuntimeModelManager();
        RuntimeModelRecord bodyRecord;
        return runtimeModelManager != null
            && runtimeModelManager.TryGetLoadedRecord(item.BodyModelKey, out bodyRecord)
            && bodyRecord != null
            && bodyRecord.RootGameObject != null
            && bodyRecord.RootGameObject.activeSelf;
    }

    private void HideEvidenceForItem(HistoryPlacementRestorationItem item)
    {
        if (item == null)
        {
            return;
        }
        item.EvidenceVisibleRequested = false;
        if (item.EvidenceImageObject != null)
        {
            item.EvidenceImageObject.SetActive(false);
        }

        ResolveRuntimeModelManager();
        RuntimeModelRecord bodyRecord;
        if (runtimeModelManager != null
            && runtimeModelManager.TryGetLoadedRecord(item.BodyModelKey, out bodyRecord)
            && bodyRecord != null
            && bodyRecord.RootGameObject != null)
        {
            bodyRecord.RootGameObject.SetActive(false);
        }
    }

    private bool CreateDisplayForPayload(string taskId, JObject result, JObject payload, Vector3 arucoPosition, Quaternion arucoRotation)
    {
        JObject display = payload != null ? payload["display"] as JObject : null;
        string status = payload != null && payload["status"] != null ? payload["status"].ToString() : result?["status"]?.ToString() ?? "";
        JObject polyhedron = display != null ? display["polyhedron"] as JObject : null;
        bool hasPolyhedronPose = false;
        Vector3 polyWorldPosition = Vector3.zero;
        Quaternion polyWorldRotation = Quaternion.identity;
        if (polyhedron != null && (polyhedron["enabled"] == null || polyhedron["enabled"].Value<bool>()))
        {
            JObject polyhedronPose = polyhedron["pose_aruco"] as JObject;
            hasPolyhedronPose = TryReadArucoPose(polyhedronPose, arucoPosition, arucoRotation, out polyWorldPosition, out polyWorldRotation);
        }
        if (!hasPolyhedronPose && !TryReadFallbackPolyhedronPose(result, arucoPosition, arucoRotation, out polyWorldPosition, out polyWorldRotation))
        {
            return false;
        }

        string shape = polyhedron != null && polyhedron["shape"] != null ? polyhedron["shape"].ToString() : FallbackShapeForStatus(status);
        float edgeLength = polyhedron != null && polyhedron["edge_length_m"] != null ? polyhedron["edge_length_m"].Value<float>() : 0.12f;
        string itemKey = string.IsNullOrEmpty(taskId) ? System.Guid.NewGuid().ToString("N") : taskId;
        GameObject polyObject = CreatePolyhedron(shape, edgeLength, status);
        polyObject.name = "HistoryPlacement_" + shape + "_" + itemKey;
        polyObject.transform.SetParent(activeRoot.transform, false);
        polyObject.transform.SetPositionAndRotation(polyWorldPosition, polyWorldRotation);

        HistoryPlacementRestorationInteractable interactable = polyObject.AddComponent<HistoryPlacementRestorationInteractable>();
        interactable.Configure(this, itemKey);
        HistoryPlacementRestorationRotator rotator = polyObject.AddComponent<HistoryPlacementRestorationRotator>();
        rotator.EulerDegreesPerSecond = ResolveRotationSpeed(shape);

        HistoryPlacementRestorationItem item = new HistoryPlacementRestorationItem
        {
            Key = itemKey,
            TaskId = taskId,
            Status = status,
            PolyhedronObject = polyObject,
            DurationSeconds = ReadAnimationDuration(display),
            DownloadModel = BuildDownloadModel(taskId, result),
            TakenRgbUrl = ReadNestedString(result, "taken_object_detection_urls", "result_rgb_url"),
            BodyFbxUrl = ReadNestedString(result, "sam3d_body_mesh_urls", "selected_person_fbx_url"),
            BodyModelKey = "body:" + itemKey,
            ArucoReference = CloneOrNull(result != null ? result["aruco_reference"] : null),
        };

        JObject animation = display != null ? display["animation"] as JObject : null;
        if (animation != null && animation["enabled"] != null && animation["enabled"].Value<bool>())
        {
            JObject fromPose = animation["from_pose_aruco"] as JObject;
            JObject toPose = animation["to_pose_aruco"] as JObject;
            if (TryReadArucoPose(fromPose, arucoPosition, arucoRotation, out Vector3 fromPosition, out Quaternion fromRotation)
                && TryReadArucoPose(toPose, arucoPosition, arucoRotation, out Vector3 toPosition, out Quaternion toRotation))
            {
                item.HasAnimation = true;
                item.FromPosition = fromPosition;
                item.FromRotation = fromRotation;
                item.ToPosition = toPosition;
                item.ToRotation = toRotation;
                item.CloneObject = CreateAnimationClone(taskId, fromPosition, fromRotation);
            }
        }

        activeItems[itemKey] = item;
        return true;
    }

    private string FallbackShapeForStatus(string status)
    {
        string normalized = (status ?? "").ToUpperInvariant();
        if (normalized == "MOVED")
        {
            return "cube";
        }
        if (normalized == "MISSING" || normalized == "TAKEN")
        {
            return "tetrahedron";
        }
        if (normalized == "STABLE")
        {
            return "octahedron";
        }
        return "icosahedron";
    }

    private bool TryReadFallbackPolyhedronPose(
        JObject result,
        Vector3 arucoPosition,
        Quaternion arucoRotation,
        out Vector3 worldPosition,
        out Quaternion worldRotation
    )
    {
        worldPosition = Vector3.zero;
        worldRotation = Quaternion.identity;
        JObject modelInstance = result != null ? result["model_instance"] as JObject : null;
        JObject spatialBox = result != null ? result["sam3_spatial_box"] as JObject : null;
        if (spatialBox == null && modelInstance != null)
        {
            spatialBox = modelInstance["sam3_spatial_box"] as JObject;
        }
        if (TryReadSpatialBoxTop(spatialBox, out worldPosition))
        {
            return true;
        }

        JObject objectAruco = result != null ? result["object_aruco"] as JObject : null;
        if (objectAruco == null && modelInstance != null)
        {
            objectAruco = modelInstance["object_aruco"] as JObject;
        }
        if (TryReadArucoPose(objectAruco, arucoPosition, arucoRotation, out worldPosition, out worldRotation))
        {
            worldPosition += Vector3.up * PolyhedronTopClearanceMeters;
            return true;
        }

        JObject objectWorld = result != null ? result["object_world"] as JObject : null;
        if (objectWorld == null && modelInstance != null)
        {
            objectWorld = modelInstance["object_world"] as JObject;
        }
        if (TryReadWorldPose(objectWorld, out worldPosition, out worldRotation))
        {
            worldPosition += Vector3.up * PolyhedronTopClearanceMeters;
            return true;
        }
        return false;
    }

    private bool TryReadSpatialBoxTop(JObject spatialBox, out Vector3 worldPosition)
    {
        worldPosition = Vector3.zero;
        if (spatialBox == null)
        {
            return false;
        }
        if (TryReadVector3(spatialBox["aabb_min_world"], out Vector3 minWorld)
            && TryReadVector3(spatialBox["aabb_max_world"], out Vector3 maxWorld))
        {
            worldPosition = new Vector3(
                (minWorld.x + maxWorld.x) * 0.5f,
                Mathf.Max(minWorld.y, maxWorld.y) + PolyhedronTopClearanceMeters,
                (minWorld.z + maxWorld.z) * 0.5f
            );
            return true;
        }
        if (TryReadVector3(spatialBox["center_world"], out Vector3 centerWorld)
            && TryReadVector3(spatialBox["size_world"], out Vector3 sizeWorld))
        {
            worldPosition = centerWorld + Vector3.up * (Mathf.Abs(sizeWorld.y) * 0.5f + PolyhedronTopClearanceMeters);
            return true;
        }
        return false;
    }

    private bool TryReadWorldPose(JObject pose, out Vector3 worldPosition, out Quaternion worldRotation)
    {
        worldPosition = Vector3.zero;
        worldRotation = Quaternion.identity;
        if (pose == null || !TryReadVector3(pose["position"], out worldPosition))
        {
            return false;
        }
        TryReadQuaternion(pose["rotation_quaternion_xyzw"], out worldRotation);
        return true;
    }

    private JObject BuildDownloadModel(string taskId, JObject result)
    {
        JObject modelInstance = result != null ? result["model_instance"] as JObject : null;
        if (modelInstance == null)
        {
            return null;
        }

        JObject downloadModel = new JObject
        {
            ["task_id"] = string.IsNullOrEmpty(taskId) ? modelInstance["task_id"]?.ToString() ?? "" : taskId,
            ["model_instance"] = modelInstance.DeepClone(),
        };
        foreach (string key in new[] { "object_world", "object_aruco", "aruco_reference", "sam3_spatial_box" })
        {
            JToken value = result[key];
            if (value != null && value.Type != JTokenType.Null)
            {
                downloadModel[key] = value.DeepClone();
            }
        }
        return downloadModel;
    }

    private bool QueueModelDownloadForItem(HistoryPlacementRestorationItem item)
    {
        if (item == null || item.DownloadQueued || item.DownloadModel == null)
        {
            return false;
        }
        ResolveRuntimeModelManager();
        if (runtimeModelManager != null && runtimeModelManager.HasModel(item.TaskId))
        {
            return false;
        }
        if (ShuJuQingQiu.initialize == null)
        {
            return false;
        }
        if (ShuJuQingQiu.initialize.DownloadRuntimeModelFromSpatialQueryModel((JObject)item.DownloadModel.DeepClone()))
        {
            item.DownloadQueued = true;
            return true;
        }
        return false;
    }

    private IEnumerator WaitForModelThenAnimate(HistoryPlacementRestorationItem item)
    {
        float timeoutAt = Time.time + 30f;
        while (item != null && Time.time < timeoutAt)
        {
            if (item.CloneObject == null)
            {
                item.CloneObject = CreateAnimationClone(item.TaskId, item.FromPosition, item.FromRotation);
            }
            if (item.CloneObject != null)
            {
                item.DownloadQueued = false;
                StartItemAnimation(item);
                yield break;
            }
            yield return null;
        }
        if (item != null)
        {
            item.DownloadQueued = false;
            item.AnimationCoroutine = null;
        }
        ShowFrontMessage("history_placement_restoration_no_animation_model");
    }

    private bool EnsureEvidenceForItem(HistoryPlacementRestorationItem item)
    {
        if (item == null)
        {
            return false;
        }

        item.EvidenceVisibleRequested = true;
        EnsureActiveRoot();
        bool handled = false;
        if (EnsureTakenImageForItem(item))
        {
            handled = true;
        }
        if (EnsureBodyMeshForItem(item))
        {
            handled = true;
        }
        return handled;
    }

    private bool EnsureTakenImageForItem(HistoryPlacementRestorationItem item)
    {
        if (item == null)
        {
            return false;
        }
        if (item.EvidenceImageObject != null)
        {
            item.EvidenceImageObject.SetActive(item.EvidenceVisibleRequested);
            return true;
        }
        if (item.ImageRequestInFlight)
        {
            return true;
        }
        if (string.IsNullOrEmpty(item.TakenRgbUrl))
        {
            return false;
        }

        item.ImageRequestInFlight = true;
        var request = new HTTPRequest(new Uri(item.TakenRgbUrl), HTTPMethods.Get, OnTakenEvidenceImageDownloaded);
        request.Tag = item.Key;
        Debug.Log("[HTTP][REQ] history-evidence-taken-rgb url=" + item.TakenRgbUrl);
        request.Send();
        ShowFrontMessage("history_placement_restoration_loading_taken_rgb");
        return true;
    }

    private void OnTakenEvidenceImageDownloaded(HTTPRequest request, HTTPResponse response)
    {
        string itemKey = request != null ? request.Tag as string : "";
        if (string.IsNullOrEmpty(itemKey) || !activeItems.ContainsKey(itemKey))
        {
            return;
        }

        HistoryPlacementRestorationItem item = activeItems[itemKey];
        item.ImageRequestInFlight = false;
        string statusCode = response != null ? response.StatusCode.ToString() : "no_response";
        int byteCount = response != null && response.Data != null ? response.Data.Length : 0;
        Debug.Log("[HTTP][RESP] history-evidence-taken-rgb status=" + statusCode + " success=" + (response != null && response.IsSuccess).ToString() + " bytes=" + byteCount.ToString());
        if (response == null || !response.IsSuccess || response.Data == null || response.Data.Length == 0)
        {
            ShowFrontMessage("history_placement_restoration_ERR_taken_rgb");
            return;
        }

        Texture2D texture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        if (!texture.LoadImage(response.Data))
        {
            Destroy(texture);
            ShowFrontMessage("history_placement_restoration_ERR_taken_rgb");
            return;
        }

        if (item.EvidenceTexture != null)
        {
            Destroy(item.EvidenceTexture);
        }
        item.EvidenceTexture = texture;
        item.EvidenceImageObject = CreateEvidenceImageQuad(item, texture);
        if (item.EvidenceImageObject != null)
        {
            item.EvidenceImageObject.SetActive(item.EvidenceVisibleRequested);
            UpdateEvidenceImagePlacement(item, true);
        }
    }

    private GameObject CreateEvidenceImageQuad(HistoryPlacementRestorationItem item, Texture2D texture)
    {
        if (item == null || texture == null)
        {
            return null;
        }
        EnsureActiveRoot();
        if (activeRoot == null)
        {
            return null;
        }

        GameObject quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.name = "HistoryPlacementTakenRgb_" + item.Key;
        quad.transform.SetParent(activeRoot.transform, false);
        Collider collider = quad.GetComponent<Collider>();
        if (collider != null)
        {
            Destroy(collider);
        }

        float aspect = texture.height > 0 ? (float)texture.width / (float)texture.height : 1.0f;
        float height = 0.24f;
        quad.transform.localScale = new Vector3(height * aspect, height, 1.0f);

        Renderer renderer = quad.GetComponent<Renderer>();
        if (renderer != null)
        {
            Shader shader = Shader.Find("Unlit/Texture");
            Material material = shader != null ? new Material(shader) : new Material(Shader.Find("Standard"));
            material.mainTexture = texture;
            if (material.HasProperty("_Cull"))
            {
                material.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            }
            renderer.material = material;
        }
        UpdateEvidenceImagePlacement(item, true);
        return quad;
    }

    private void UpdateEvidenceImagePlacement(HistoryPlacementRestorationItem item, bool force)
    {
        if (item == null || item.EvidenceImageObject == null)
        {
            return;
        }

        Vector3 anchor = ResolveEvidenceAnchorPosition(item);
        Camera camera = Camera.main;
        Vector3 right = camera != null ? camera.transform.right : Vector3.right;
        Vector3 targetPosition = anchor + Vector3.up * EvidenceImageVerticalOffsetMeters + right * EvidenceImageSideOffsetMeters;
        item.EvidenceImageObject.transform.position = force
            ? targetPosition
            : Vector3.Lerp(item.EvidenceImageObject.transform.position, targetPosition, Time.deltaTime * EvidenceImageLerpSpeed);

        if (camera != null)
        {
            Vector3 toCamera = item.EvidenceImageObject.transform.position - camera.transform.position;
            if (toCamera.sqrMagnitude > 0.0001f)
            {
                item.EvidenceImageObject.transform.rotation = Quaternion.LookRotation(toCamera.normalized, Vector3.up);
            }
        }
    }

    private Vector3 ResolveEvidenceAnchorPosition(HistoryPlacementRestorationItem item)
    {
        ResolveRuntimeModelManager();
        RuntimeModelRecord record;
        if (runtimeModelManager != null)
        {
            if (!string.IsNullOrEmpty(item.TaskId) && runtimeModelManager.TryGetLoadedRecord(item.TaskId, out record))
            {
                if (TryGetRecordTopCenter(record, out Vector3 topCenter))
                {
                    return topCenter;
                }
            }
            if (!string.IsNullOrEmpty(item.ModelKey) && runtimeModelManager.TryGetLoadedRecord(item.ModelKey, out record))
            {
                if (TryGetRecordTopCenter(record, out Vector3 topCenter))
                {
                    return topCenter;
                }
            }
        }

        if (item.PolyhedronObject != null)
        {
            return item.PolyhedronObject.transform.position;
        }
        return item.ToPosition;
    }

    private bool TryGetRecordTopCenter(RuntimeModelRecord record, out Vector3 topCenter)
    {
        topCenter = Vector3.zero;
        if (record == null)
        {
            return false;
        }
        if (record.RootGameObject != null && TryGetRendererBounds(record.RootGameObject, out Bounds bounds))
        {
            topCenter = new Vector3(bounds.center.x, bounds.max.y, bounds.center.z);
            return true;
        }
        if (record.SpatialBox != null && record.SpatialBox.IsReady)
        {
            topCenter = new Vector3(record.SpatialBox.CenterWorld.x, record.SpatialBox.AabbMaxWorld.y, record.SpatialBox.CenterWorld.z);
            return true;
        }
        return false;
    }

    private bool TryGetRendererBounds(GameObject root, out Bounds bounds)
    {
        bounds = new Bounds(Vector3.zero, Vector3.zero);
        if (root == null)
        {
            return false;
        }
        bool initialized = false;
        foreach (Renderer renderer in root.GetComponentsInChildren<Renderer>(true))
        {
            if (renderer == null)
            {
                continue;
            }
            if (!initialized)
            {
                bounds = renderer.bounds;
                initialized = true;
            }
            else
            {
                bounds.Encapsulate(renderer.bounds);
            }
        }
        return initialized;
    }

    private bool EnsureBodyMeshForItem(HistoryPlacementRestorationItem item)
    {
        if (item == null || string.IsNullOrEmpty(item.BodyFbxUrl))
        {
            return false;
        }
        if (item.BodyDownloadQueued)
        {
            return true;
        }

        ResolveRuntimeModelManager();
        RuntimeModelRecord bodyRecord;
        if (runtimeModelManager != null && runtimeModelManager.TryGetLoadedRecord(item.BodyModelKey, out bodyRecord))
        {
            if (bodyRecord != null && bodyRecord.RootGameObject != null)
            {
                ApplyEvidenceBodyMaterial(bodyRecord.RootGameObject);
                bodyRecord.RootGameObject.SetActive(item.EvidenceVisibleRequested);
            }
            return true;
        }
        if (ShuJuQingQiu.initialize == null)
        {
            return false;
        }

        JObject bodyRootPose = BuildBodyRootArucoPose(item);
        JObject modelInstance = new JObject
        {
            ["model_key"] = item.BodyModelKey,
            ["task_id"] = item.BodyModelKey,
            ["fbx_url"] = item.BodyFbxUrl,
            ["is_evidence_overlay"] = true,
            ["object_aruco"] = bodyRootPose.DeepClone(),
        };
        if (item.ArucoReference != null)
        {
            modelInstance["aruco_reference"] = item.ArucoReference.DeepClone();
        }

        JObject bodyModel = new JObject
        {
            ["task_id"] = item.BodyModelKey,
            ["is_evidence_overlay"] = true,
            ["model_instance"] = modelInstance,
            ["object_aruco"] = bodyRootPose,
        };
        if (item.ArucoReference != null)
        {
            bodyModel["aruco_reference"] = item.ArucoReference.DeepClone();
        }

        if (ShuJuQingQiu.initialize.DownloadRuntimeModelFromSpatialQueryModel(bodyModel))
        {
            item.BodyDownloadQueued = true;
            if (item.BodyVisibilityCoroutine != null)
            {
                StopCoroutine(item.BodyVisibilityCoroutine);
            }
            item.BodyVisibilityCoroutine = StartCoroutine(WaitForBodyMeshThenApplyVisibility(item));
            ShowFrontMessage("history_placement_restoration_loading_body_mesh");
            return true;
        }
        return false;
    }

    private JObject BuildBodyRootArucoPose(HistoryPlacementRestorationItem item)
    {
        Vector3 rootPosition = Vector3.zero;
        JObject objectAruco = item != null ? item.ObjectAruco as JObject : null;
        if (item != null
            && item.HasBodyObjectCenterAruco
            && objectAruco != null
            && TryReadVector3(objectAruco["position"], out Vector3 objectCenterAruco))
        {
            rootPosition = objectCenterAruco - item.BodyObjectCenterAruco;
        }

        return new JObject
        {
            ["position"] = new JArray(rootPosition.x, rootPosition.y, rootPosition.z),
            ["rotation_quaternion_xyzw"] = new JArray(0.0f, 0.0f, 0.0f, 1.0f),
        };
    }

    private IEnumerator WaitForBodyMeshThenApplyVisibility(HistoryPlacementRestorationItem item)
    {
        float timeoutAt = Time.time + 30f;
        while (item != null && Time.time < timeoutAt)
        {
            ResolveRuntimeModelManager();
            RuntimeModelRecord bodyRecord;
            if (runtimeModelManager != null
                && runtimeModelManager.TryGetLoadedRecord(item.BodyModelKey, out bodyRecord)
                && bodyRecord != null
                && bodyRecord.RootGameObject != null)
            {
                ApplyEvidenceBodyMaterial(bodyRecord.RootGameObject);
                bodyRecord.RootGameObject.SetActive(item.EvidenceVisibleRequested);
                item.BodyDownloadQueued = false;
                item.BodyVisibilityCoroutine = null;
                yield break;
            }
            yield return null;
        }
        if (item != null)
        {
            item.BodyDownloadQueued = false;
            item.BodyVisibilityCoroutine = null;
        }
    }

    private void ApplyEvidenceBodyMaterial(GameObject root)
    {
        if (root == null)
        {
            return;
        }

        Material material = GetEvidenceBodyMaterial();
        foreach (Renderer renderer in root.GetComponentsInChildren<Renderer>(true))
        {
            if (renderer != null)
            {
                renderer.sharedMaterial = material;
            }
        }
    }

    private Material GetEvidenceBodyMaterial()
    {
        if (evidenceBodyMaterial == null)
        {
            evidenceBodyMaterial = BuildMaterial(new Color(0.62f, 0.66f, 0.70f, 0.38f));
        }
        return evidenceBodyMaterial;
    }

    private static void DestroyMaterial(Material material)
    {
        if (material != null)
        {
            Destroy(material);
        }
    }

    private static string ReadNestedString(JObject payload, string objectKey, string valueKey)
    {
        JObject obj = payload != null ? payload[objectKey] as JObject : null;
        return obj != null ? obj[valueKey]?.ToString() ?? "" : "";
    }

    private static JToken CloneOrNull(JToken token)
    {
        return token != null && token.Type != JTokenType.Null ? token.DeepClone() : null;
    }

    private GameObject CreatePolyhedron(string shape, float edgeLength, string status)
    {
        float safeEdge = Mathf.Max(0.02f, edgeLength);
        Material material = ResolveMaterial(shape, status);
        if (shape == "tetrahedron")
        {
            return CreateTetrahedron(safeEdge, material);
        }
        if (shape == "octahedron")
        {
            return CreateOctahedron(safeEdge, material);
        }
        if (shape == "dodecahedron")
        {
            return CreateDodecahedron(safeEdge, material);
        }
        if (shape == "icosahedron")
        {
            return CreateIcosahedron(safeEdge, material);
        }

        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.transform.localScale = Vector3.one * safeEdge;
        Renderer renderer = cube.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material = material;
        }
        return cube;
    }

    private GameObject CreateTetrahedron(float edgeLength, Material material)
    {
        float scale = edgeLength / Mathf.Sqrt(8f);
        Vector3[] vertices =
        {
            new Vector3(1f, 1f, 1f) * scale,
            new Vector3(1f, -1f, -1f) * scale,
            new Vector3(-1f, 1f, -1f) * scale,
            new Vector3(-1f, -1f, 1f) * scale,
        };
        int[] triangles =
        {
            0, 2, 1,
            0, 1, 3,
            0, 3, 2,
            1, 2, 3,
        };
        return CreateMeshPolyhedron("HistoryPlacementTetrahedron", vertices, triangles, material);
    }

    private GameObject CreateOctahedron(float edgeLength, Material material)
    {
        float radius = edgeLength / Mathf.Sqrt(2f);
        Vector3[] vertices =
        {
            new Vector3(0f, radius, 0f),
            new Vector3(radius, 0f, 0f),
            new Vector3(0f, 0f, radius),
            new Vector3(-radius, 0f, 0f),
            new Vector3(0f, 0f, -radius),
            new Vector3(0f, -radius, 0f),
        };
        int[] triangles =
        {
            0, 1, 2,
            0, 2, 3,
            0, 3, 4,
            0, 4, 1,
            5, 2, 1,
            5, 3, 2,
            5, 4, 3,
            5, 1, 4,
        };
        return CreateMeshPolyhedron("HistoryPlacementOctahedron", vertices, triangles, material);
    }

    private GameObject CreateIcosahedron(float edgeLength, Material material)
    {
        float phi = (1f + Mathf.Sqrt(5f)) * 0.5f;
        float scale = edgeLength * 0.5f;
        Vector3[] vertices =
        {
            new Vector3(-1f, phi, 0f) * scale,
            new Vector3(1f, phi, 0f) * scale,
            new Vector3(-1f, -phi, 0f) * scale,
            new Vector3(1f, -phi, 0f) * scale,
            new Vector3(0f, -1f, phi) * scale,
            new Vector3(0f, 1f, phi) * scale,
            new Vector3(0f, -1f, -phi) * scale,
            new Vector3(0f, 1f, -phi) * scale,
            new Vector3(phi, 0f, -1f) * scale,
            new Vector3(phi, 0f, 1f) * scale,
            new Vector3(-phi, 0f, -1f) * scale,
            new Vector3(-phi, 0f, 1f) * scale,
        };
        int[] triangles = IcosahedronTriangles();
        return CreateMeshPolyhedron("HistoryPlacementIcosahedron", vertices, triangles, material);
    }

    private GameObject CreateDodecahedron(float edgeLength, Material material)
    {
        float phi = (1f + Mathf.Sqrt(5f)) * 0.5f;
        Vector3[] icoVertices =
        {
            new Vector3(-1f, phi, 0f),
            new Vector3(1f, phi, 0f),
            new Vector3(-1f, -phi, 0f),
            new Vector3(1f, -phi, 0f),
            new Vector3(0f, -1f, phi),
            new Vector3(0f, 1f, phi),
            new Vector3(0f, -1f, -phi),
            new Vector3(0f, 1f, -phi),
            new Vector3(phi, 0f, -1f),
            new Vector3(phi, 0f, 1f),
            new Vector3(-phi, 0f, -1f),
            new Vector3(-phi, 0f, 1f),
        };
        int[] icoTriangles = IcosahedronTriangles();
        List<Vector3> dodecaVertices = new List<Vector3>();
        for (int i = 0; i < icoTriangles.Length; i += 3)
        {
            Vector3 center = (icoVertices[icoTriangles[i]] + icoVertices[icoTriangles[i + 1]] + icoVertices[icoTriangles[i + 2]]) / 3f;
            dodecaVertices.Add(center.normalized);
        }

        List<int> triangles = new List<int>();
        for (int vertexIndex = 0; vertexIndex < icoVertices.Length; vertexIndex++)
        {
            List<int> faceIndices = new List<int>();
            for (int tri = 0; tri < icoTriangles.Length / 3; tri++)
            {
                int a = icoTriangles[tri * 3];
                int b = icoTriangles[tri * 3 + 1];
                int c = icoTriangles[tri * 3 + 2];
                if (a == vertexIndex || b == vertexIndex || c == vertexIndex)
                {
                    faceIndices.Add(tri);
                }
            }
            SortFaceAroundNormal(faceIndices, dodecaVertices, icoVertices[vertexIndex].normalized);
            for (int i = 1; i + 1 < faceIndices.Count; i++)
            {
                AddOrientedTriangle(triangles, dodecaVertices, faceIndices[0], faceIndices[i], faceIndices[i + 1]);
            }
        }

        Vector3[] vertices = dodecaVertices.ToArray();
        NormalizeEdgeLength(vertices, edgeLength);
        return CreateMeshPolyhedron("HistoryPlacementDodecahedron", vertices, triangles.ToArray(), material);
    }

    private static int[] IcosahedronTriangles()
    {
        return new int[]
        {
            0, 11, 5,
            0, 5, 1,
            0, 1, 7,
            0, 7, 10,
            0, 10, 11,
            1, 5, 9,
            5, 11, 4,
            11, 10, 2,
            10, 7, 6,
            7, 1, 8,
            3, 9, 4,
            3, 4, 2,
            3, 2, 6,
            3, 6, 8,
            3, 8, 9,
            4, 9, 5,
            2, 4, 11,
            6, 2, 10,
            8, 6, 7,
            9, 8, 1,
        };
    }

    private static void SortFaceAroundNormal(List<int> faceIndices, List<Vector3> vertices, Vector3 normal)
    {
        Vector3 axis = Mathf.Abs(Vector3.Dot(normal, Vector3.up)) > 0.9f ? Vector3.right : Vector3.up;
        Vector3 tangent = Vector3.Cross(normal, axis).normalized;
        Vector3 bitangent = Vector3.Cross(normal, tangent).normalized;
        faceIndices.Sort((a, b) =>
        {
            Vector3 va = Vector3.ProjectOnPlane(vertices[a], normal).normalized;
            Vector3 vb = Vector3.ProjectOnPlane(vertices[b], normal).normalized;
            float angleA = Mathf.Atan2(Vector3.Dot(va, bitangent), Vector3.Dot(va, tangent));
            float angleB = Mathf.Atan2(Vector3.Dot(vb, bitangent), Vector3.Dot(vb, tangent));
            return angleA.CompareTo(angleB);
        });
    }

    private static void AddOrientedTriangle(List<int> triangles, List<Vector3> vertices, int a, int b, int c)
    {
        Vector3 normal = Vector3.Cross(vertices[b] - vertices[a], vertices[c] - vertices[a]);
        Vector3 center = (vertices[a] + vertices[b] + vertices[c]) / 3f;
        if (Vector3.Dot(normal, center) < 0f)
        {
            triangles.Add(a);
            triangles.Add(c);
            triangles.Add(b);
        }
        else
        {
            triangles.Add(a);
            triangles.Add(b);
            triangles.Add(c);
        }
    }

    private static void NormalizeEdgeLength(Vector3[] vertices, float edgeLength)
    {
        float minDistance = float.MaxValue;
        for (int i = 0; i < vertices.Length; i++)
        {
            for (int j = i + 1; j < vertices.Length; j++)
            {
                float distance = Vector3.Distance(vertices[i], vertices[j]);
                if (distance > 0.0001f && distance < minDistance)
                {
                    minDistance = distance;
                }
            }
        }
        if (minDistance <= 0.0001f || minDistance == float.MaxValue)
        {
            return;
        }
        float scale = edgeLength / minDistance;
        for (int i = 0; i < vertices.Length; i++)
        {
            vertices[i] *= scale;
        }
    }

    private GameObject CreateMeshPolyhedron(string name, Vector3[] vertices, int[] triangles, Material material)
    {
        Mesh mesh = new Mesh();
        mesh.name = name + "Mesh";
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        GameObject obj = new GameObject(name);
        MeshFilter filter = obj.AddComponent<MeshFilter>();
        filter.sharedMesh = mesh;
        MeshRenderer renderer = obj.AddComponent<MeshRenderer>();
        renderer.material = material;
        MeshCollider collider = obj.AddComponent<MeshCollider>();
        collider.sharedMesh = mesh;
        collider.convex = true;
        return obj;
    }

    private Vector3 ResolveRotationSpeed(string shape)
    {
        if (shape == "tetrahedron")
        {
            return new Vector3(65f, 95f, 30f);
        }
        if (shape == "octahedron")
        {
            return new Vector3(0f, 75f, 45f);
        }
        if (shape == "dodecahedron")
        {
            return new Vector3(30f, 70f, 45f);
        }
        if (shape == "icosahedron")
        {
            return new Vector3(55f, 40f, 80f);
        }
        return new Vector3(0f, 90f, 0f);
    }

    private Material ResolveMaterial(string shape, string status)
    {
        if (shape == "tetrahedron")
        {
            if (tetrahedronMaterial == null)
            {
                tetrahedronMaterial = BuildMaterial(new Color(0.2f, 0.85f, 0.35f, 0.82f));
            }
            return tetrahedronMaterial;
        }
        if (shape == "octahedron")
        {
            if (octahedronMaterial == null)
            {
                octahedronMaterial = BuildMaterial(new Color(0.4f, 0.55f, 1f, 0.82f));
            }
            return octahedronMaterial;
        }
        if (shape == "dodecahedron")
        {
            if (dodecahedronMaterial == null)
            {
                dodecahedronMaterial = BuildMaterial(new Color(1f, 0.72f, 0.2f, 0.82f));
            }
            return dodecahedronMaterial;
        }
        if (shape == "icosahedron")
        {
            if (icosahedronMaterial == null)
            {
                icosahedronMaterial = BuildMaterial(new Color(0.9f, 0.35f, 1f, 0.82f));
            }
            return icosahedronMaterial;
        }

        if (cubeMaterial == null)
        {
            cubeMaterial = BuildMaterial(new Color(0.0f, 0.35f, 1f, 0.82f));
        }
        return cubeMaterial;
    }

    private Material BuildMaterial(Color color)
    {
        Shader shader = Shader.Find("Standard");
        Material material = shader != null ? new Material(shader) : new Material(Shader.Find("Sprites/Default"));
        material.color = color;
        if (material.HasProperty("_Mode"))
        {
            material.SetFloat("_Mode", 3f);
            material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            material.SetInt("_ZWrite", 0);
            material.DisableKeyword("_ALPHATEST_ON");
            material.EnableKeyword("_ALPHABLEND_ON");
            material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            material.renderQueue = 3000;
        }
        return material;
    }

    private GameObject CreateAnimationClone(string taskId, Vector3 fromPosition, Quaternion fromRotation)
    {
        RuntimeModelRecord record;
        if (runtimeModelManager == null || !runtimeModelManager.TryGetLoadedRecord(taskId, out record))
        {
            Debug.LogWarning("[HistoryPlacementRestoration] Model is not loaded; animation clone skipped for task " + taskId);
            return null;
        }
        if (record == null || record.RootGameObject == null)
        {
            return null;
        }

        GameObject clone = Instantiate(record.RootGameObject, fromPosition, fromRotation, activeRoot.transform);
        clone.name = "HistoryPlacementAnimationClone_" + taskId;
        foreach (RuntimeModelEventIdentity identity in clone.GetComponentsInChildren<RuntimeModelEventIdentity>(true))
        {
            Destroy(identity);
        }
        foreach (HistoryPlacementRestorationInteractable interactable in clone.GetComponentsInChildren<HistoryPlacementRestorationInteractable>(true))
        {
            Destroy(interactable);
        }
        clone.SetActive(false);
        return clone;
    }

    private IEnumerator AnimateCloneToOriginal(HistoryPlacementRestorationItem item)
    {
        float duration = Mathf.Max(0.05f, item.DurationSeconds);
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / duration);
            t = t * t * (3f - 2f * t);
            item.CloneObject.transform.SetPositionAndRotation(
                Vector3.Lerp(item.FromPosition, item.ToPosition, t),
                Quaternion.Slerp(item.FromRotation, item.ToRotation, t)
            );
            yield return null;
        }
        item.CloneObject.transform.SetPositionAndRotation(item.ToPosition, item.ToRotation);
        item.AnimationCoroutine = null;
    }

    private float ReadAnimationDuration(JObject display)
    {
        JObject animation = display != null ? display["animation"] as JObject : null;
        if (animation != null && animation["duration_seconds"] != null)
        {
            return Mathf.Max(0.05f, animation["duration_seconds"].Value<float>());
        }
        return 1.2f;
    }

    private bool TryReadArucoPose(
        JObject pose,
        Vector3 arucoPosition,
        Quaternion arucoRotation,
        out Vector3 worldPosition,
        out Quaternion worldRotation
    )
    {
        worldPosition = Vector3.zero;
        worldRotation = Quaternion.identity;
        if (pose == null)
        {
            return false;
        }
        if (!TryReadVector3(pose["position"], out Vector3 localPosition))
        {
            return false;
        }
        Quaternion localRotation = Quaternion.identity;
        TryReadQuaternion(pose["rotation_quaternion_xyzw"], out localRotation);
        worldPosition = arucoPosition + (arucoRotation * localPosition);
        worldRotation = arucoRotation * localRotation;
        return true;
    }

    private bool TryGetArucoReference(out Vector3 position, out Quaternion rotation)
    {
        position = Vector3.zero;
        rotation = Quaternion.identity;
        if (runtimeModelManager != null && runtimeModelManager.TryGetCurrentArucoReference(out position, out rotation))
        {
            return true;
        }

        ShuJuQingQiu loader = ShuJuQingQiu.initialize != null ? ShuJuQingQiu.initialize : FindObjectOfType<ShuJuQingQiu>();
        if (loader != null && loader.hasArucoReferencePose)
        {
            position = loader.arucoReferencePosition;
            rotation = loader.arucoReferenceRotation;
            return true;
        }
        return false;
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

    private bool TryReadQuaternion(JToken token, out Quaternion value)
    {
        value = Quaternion.identity;
        JArray arr = token as JArray;
        if (arr == null || arr.Count < 4)
        {
            return false;
        }
        value = new Quaternion(arr[0].Value<float>(), arr[1].Value<float>(), arr[2].Value<float>(), arr[3].Value<float>());
        return true;
    }

    private void ResolveRuntimeModelManager()
    {
        if (runtimeModelManager == null)
        {
            runtimeModelManager = RuntimeModelManager.Instance;
        }
    }

    private void EnsureActiveRoot()
    {
        if (activeRoot == null)
        {
            activeRoot = new GameObject(RootName);
        }
    }

    private void ShowFrontMessage(string message)
    {
        if (Game_M.initialize != null)
        {
            Game_M.initialize.XianShi(message);
        }
    }

    private class HistoryPlacementRestorationItem
    {
        public string Key = "";
        public string TaskId = "";
        public string ModelKey = "";
        public string Status = "";
        public GameObject PolyhedronObject;
        public GameObject CloneObject;
        public bool HasAnimation;
        public Vector3 FromPosition;
        public Quaternion FromRotation = Quaternion.identity;
        public Vector3 ToPosition;
        public Quaternion ToRotation = Quaternion.identity;
        public float DurationSeconds = 1.2f;
        public Coroutine AnimationCoroutine;
        public JObject DownloadModel;
        public bool DownloadQueued;
        public string TakenRgbUrl = "";
        public string BodyFbxUrl = "";
        public string BodyModelKey = "";
        public JToken ArucoReference;
        public JToken ObjectAruco;
        public bool HasBodyObjectCenterAruco;
        public Vector3 BodyObjectCenterAruco;
        public bool ImageRequestInFlight;
        public bool BodyDownloadQueued;
        public bool EvidenceVisibleRequested;
        public Coroutine BodyVisibilityCoroutine;
        public Texture2D EvidenceTexture;
        public GameObject EvidenceImageObject;
    }
}

public class HistoryPlacementRestorationInteractable : MonoBehaviour, IMixedRealityPointerHandler
{
    private HistoryPlacementRestorationDisplay owner;
    private string itemKey = "";

    public void Configure(HistoryPlacementRestorationDisplay newOwner, string newItemKey)
    {
        owner = newOwner;
        itemKey = newItemKey ?? "";
    }

    public void OnPointerClicked(MixedRealityPointerEventData eventData)
    {
        if (owner != null)
        {
            owner.HandlePolyhedronClicked(itemKey);
        }
        if (eventData != null)
        {
            eventData.Use();
        }
    }

    public void OnPointerDown(MixedRealityPointerEventData eventData)
    {
    }

    public void OnPointerDragged(MixedRealityPointerEventData eventData)
    {
    }

    public void OnPointerUp(MixedRealityPointerEventData eventData)
    {
    }
}

public class HistoryPlacementRestorationRotator : MonoBehaviour
{
    public Vector3 EulerDegreesPerSecond = new Vector3(0f, 90f, 0f);

    private void Update()
    {
        transform.Rotate(EulerDegreesPerSecond * Time.deltaTime, Space.Self);
    }
}
