using Microsoft.MixedReality.Toolkit.Input;
using Newtonsoft.Json.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class HistoryPlacementRestorationDisplay : MonoBehaviour
{
    private const string RootName = "HistoryPlacementRestorationDisplayRoot";

    private static HistoryPlacementRestorationDisplay _instance;
    private readonly Dictionary<string, HistoryPlacementRestorationItem> activeItems = new Dictionary<string, HistoryPlacementRestorationItem>();
    private GameObject activeRoot;
    private Material cubeMaterial;
    private Material octahedronMaterial;
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

    private void OnDestroy()
    {
        if (_instance == this)
        {
            _instance = null;
        }
        Clear();
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

            if (CreateDisplayForPayload(taskId, payload, arucoPosition, arucoRotation))
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
        if (item == null || !item.HasAnimation)
        {
            ShowFrontMessage("history_placement_restoration_no_animation_model");
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
            ShowFrontMessage("history_placement_restoration_no_animation_model");
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

    private bool CreateDisplayForPayload(string taskId, JObject payload, Vector3 arucoPosition, Quaternion arucoRotation)
    {
        JObject display = payload["display"] as JObject;
        if (display == null)
        {
            return false;
        }

        string status = payload["status"] != null ? payload["status"].ToString() : "";
        JObject polyhedron = display["polyhedron"] as JObject;
        bool polyhedronEnabled = polyhedron != null && polyhedron["enabled"] != null && polyhedron["enabled"].Value<bool>();
        if (!polyhedronEnabled)
        {
            return false;
        }

        JObject polyhedronPose = polyhedron["pose_aruco"] as JObject;
        if (!TryReadArucoPose(polyhedronPose, arucoPosition, arucoRotation, out Vector3 polyWorldPosition, out Quaternion polyWorldRotation))
        {
            return false;
        }

        string shape = polyhedron["shape"] != null ? polyhedron["shape"].ToString() : "cube";
        float edgeLength = polyhedron["edge_length_m"] != null ? polyhedron["edge_length_m"].Value<float>() : 0.1f;
        string itemKey = string.IsNullOrEmpty(taskId) ? System.Guid.NewGuid().ToString("N") : taskId;
        GameObject polyObject = CreatePolyhedron(shape, edgeLength, status);
        polyObject.name = "HistoryPlacement_" + shape + "_" + itemKey;
        polyObject.transform.SetParent(activeRoot.transform, false);
        polyObject.transform.SetPositionAndRotation(polyWorldPosition, polyWorldRotation);

        HistoryPlacementRestorationInteractable interactable = polyObject.AddComponent<HistoryPlacementRestorationInteractable>();
        interactable.Configure(this, itemKey);
        HistoryPlacementRestorationRotator rotator = polyObject.AddComponent<HistoryPlacementRestorationRotator>();
        rotator.EulerDegreesPerSecond = shape == "octahedron" ? new Vector3(0f, 75f, 45f) : new Vector3(0f, 90f, 0f);

        HistoryPlacementRestorationItem item = new HistoryPlacementRestorationItem
        {
            Key = itemKey,
            TaskId = taskId,
            Status = status,
            PolyhedronObject = polyObject,
            DurationSeconds = ReadAnimationDuration(display),
        };

        JObject animation = display["animation"] as JObject;
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

    private GameObject CreatePolyhedron(string shape, float edgeLength, string status)
    {
        if (shape == "octahedron")
        {
            return CreateOctahedron(Mathf.Max(0.02f, edgeLength), ResolveMaterial(shape, status));
        }

        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.transform.localScale = Vector3.one * Mathf.Max(0.02f, edgeLength);
        Renderer renderer = cube.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material = ResolveMaterial("cube", status);
        }
        return cube;
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
        Mesh mesh = new Mesh();
        mesh.name = "HistoryPlacementOctahedronMesh";
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        GameObject obj = new GameObject("HistoryPlacementOctahedron");
        MeshFilter filter = obj.AddComponent<MeshFilter>();
        filter.sharedMesh = mesh;
        MeshRenderer renderer = obj.AddComponent<MeshRenderer>();
        renderer.material = material;
        MeshCollider collider = obj.AddComponent<MeshCollider>();
        collider.sharedMesh = mesh;
        collider.convex = true;
        return obj;
    }

    private Material ResolveMaterial(string shape, string status)
    {
        if (shape == "octahedron")
        {
            if (octahedronMaterial == null)
            {
                octahedronMaterial = BuildMaterial(new Color(0.4f, 0.55f, 1f, 0.82f));
            }
            return octahedronMaterial;
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
