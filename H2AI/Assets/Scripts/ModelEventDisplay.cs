using BestHTTP;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

[DisallowMultipleComponent]
public class ModelEventDisplay : MonoBehaviour
{
    [SerializeField] private string serverBaseUrl = "http://10.40.1.122:7355";
    [SerializeField] private Camera targetCamera;
    [SerializeField, Min(0.01f)] private float popupDistanceFromBoxMeters = 0.15f;
    [SerializeField, Min(0.05f)] private float panelWidthMeters = 0.32f;
    [SerializeField] private Material imageMaterialTemplate;
    [SerializeField] private Material skeletonLineMaterial;
    [SerializeField] private Material bodyMeshMaterialTemplate;
    [SerializeField] private Color imageColor = Color.white;
    [SerializeField] private Color skeletonLineColor = new Color(0.1f, 0.9f, 1.0f, 1f);
    [SerializeField] private Color wristColor = new Color(1.0f, 0.75f, 0.15f, 1f);
    [SerializeField] private Color bodyMeshColor = new Color(0.55f, 0.55f, 0.55f, 0.38f);
    [SerializeField, Min(0.0005f)] private float skeletonLineWidth = 0.003f;
    [SerializeField, Min(0.002f)] private float jointPointSizeMeters = 0.01f;
    [SerializeField, Range(0f, 1f)] private float minJointScore = 0f;

    private static ModelEventDisplay _instance;

    private readonly Dictionary<string, CachedEventData> eventCache = new Dictionary<string, CachedEventData>();
    private GameObject activePopup;
    private string activeTaskId = "";
    private int requestGeneration = 0;
    private Material runtimeImageMaterial;
    private Material runtimeSkeletonMaterial;
    private Material runtimeWristMaterial;
    private Material runtimeBodyMeshMaterial;

    private static readonly string[,] SkeletonPairs =
    {
        { "nose", "neck" },
        { "neck", "left_shoulder" },
        { "left_shoulder", "left_elbow" },
        { "left_elbow", "left_wrist" },
        { "neck", "right_shoulder" },
        { "right_shoulder", "right_elbow" },
        { "right_elbow", "right_wrist" },
        { "neck", "left_hip" },
        { "left_hip", "left_knee" },
        { "left_knee", "left_ankle" },
        { "neck", "right_hip" },
        { "right_hip", "right_knee" },
        { "right_knee", "right_ankle" },
        { "left_shoulder", "right_shoulder" },
        { "left_hip", "right_hip" },
    };

    public static ModelEventDisplay Instance
    {
        get
        {
            if (_instance != null)
            {
                return _instance;
            }

            _instance = FindObjectOfType<ModelEventDisplay>();
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
        EnsureMaterials();
    }

    private void OnDestroy()
    {
        if (_instance == this)
        {
            _instance = null;
        }

        CloseAllAndClearLocalCache();
        DestroyMaterial(runtimeImageMaterial);
        DestroyMaterial(runtimeSkeletonMaterial);
        DestroyMaterial(runtimeWristMaterial);
        DestroyMaterial(runtimeBodyMeshMaterial);
    }

    public void ToggleForModel(RuntimeModelEventIdentity identity)
    {
        if (identity == null)
        {
            return;
        }

        string taskId = !string.IsNullOrEmpty(identity.TaskId) ? identity.TaskId : identity.ModelKey;
        if (string.IsNullOrEmpty(taskId))
        {
            ShowFrontMessage("model_event_ERR_missing_task_id");
            return;
        }

        if (activePopup != null && activeTaskId == taskId)
        {
            ClosePopup();
            return;
        }

        ClosePopup();

        if (eventCache.TryGetValue(taskId, out CachedEventData cached) && cached != null && cached.Texture != null)
        {
            ShowPopup(identity, cached);
            return;
        }

        RequestEvent(identity, taskId);
    }

    public void CloseAllAndClearLocalCache()
    {
        requestGeneration++;
        ClosePopup();

        foreach (CachedEventData cached in eventCache.Values)
        {
            if (cached != null && cached.Texture != null)
            {
                Destroy(cached.Texture);
            }
        }
        eventCache.Clear();
    }

    public void DeleteServerEventsForTaskIds(IEnumerable<string> taskIds)
    {
        if (taskIds == null)
        {
            return;
        }

        HashSet<string> seen = new HashSet<string>();
        foreach (string rawTaskId in taskIds)
        {
            string taskId = (rawTaskId ?? "").Trim();
            if (string.IsNullOrEmpty(taskId) || !seen.Add(taskId))
            {
                continue;
            }

            if (eventCache.TryGetValue(taskId, out CachedEventData cached) && cached != null && cached.Texture != null)
            {
                Destroy(cached.Texture);
            }
            eventCache.Remove(taskId);

            string url = NormalizeServerBaseUrl() + "/model-events/" + Uri.EscapeDataString(taskId) + "/taken-away";
            HTTPRequest request = new HTTPRequest(new Uri(url), HTTPMethods.Delete, OnDeleteEventFinished);
            request.Tag = taskId;
            request.Send();
        }
    }

    private void RequestEvent(RuntimeModelEventIdentity identity, string taskId)
    {
        int generation = ++requestGeneration;
        string url = NormalizeServerBaseUrl() + "/model-events/" + Uri.EscapeDataString(taskId) + "/taken-away";
        EventRequestContext context = new EventRequestContext
        {
            Identity = identity,
            TaskId = taskId,
            Generation = generation,
        };

        HTTPRequest request = new HTTPRequest(new Uri(url), HTTPMethods.Get, OnEventMetadataFinished);
        request.Tag = context;
        request.Send();
        ShowFrontMessage("model_event_loading");
    }

    private void OnEventMetadataFinished(HTTPRequest request, HTTPResponse response)
    {
        EventRequestContext context = request.Tag as EventRequestContext;
        if (!IsCurrentRequest(context))
        {
            return;
        }

        if (response == null || !response.IsSuccess)
        {
            string statusCode = response != null ? response.StatusCode.ToString(CultureInfo.InvariantCulture) : "no_response";
            Debug.LogWarning("[ModelEvent] Event metadata request failed: " + statusCode);
            ShowFrontMessage("model_event_none");
            return;
        }

        JObject root = JsonConvert.DeserializeObject(response.DataAsText) as JObject;
        JObject eventJ = root != null ? root["event"] as JObject : null;
        JObject urls = eventJ != null ? eventJ["download_urls"] as JObject : null;
        string rgbUrl = urls != null ? urls["rgb"]?.ToString() : "";
        string skeletonsUrl = urls != null ? urls["skeletons"]?.ToString() : "";
        string bodyMeshUrl = urls != null ? urls["body_mesh"]?.ToString() : "";
        if (string.IsNullOrEmpty(rgbUrl))
        {
            ShowFrontMessage("model_event_ERR_missing_rgb");
            return;
        }

        context.EventJson = eventJ;
        context.SkeletonsUrl = skeletonsUrl;
        context.BodyMeshUrl = bodyMeshUrl;

        HTTPRequest imageRequest = new HTTPRequest(new Uri(rgbUrl), HTTPMethods.Get, OnEventImageFinished);
        imageRequest.Tag = context;
        imageRequest.Send();
    }

    private void OnEventImageFinished(HTTPRequest request, HTTPResponse response)
    {
        EventRequestContext context = request.Tag as EventRequestContext;
        if (!IsCurrentRequest(context))
        {
            return;
        }

        if (response == null || !response.IsSuccess || response.Data == null || response.Data.Length == 0)
        {
            Debug.LogWarning("[ModelEvent] Event image request failed.");
            ShowFrontMessage("model_event_ERR_image");
            return;
        }

        Texture2D texture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        if (!texture.LoadImage(response.Data))
        {
            Destroy(texture);
            ShowFrontMessage("model_event_ERR_image_decode");
            return;
        }

        CachedEventData cached = new CachedEventData
        {
            Texture = texture,
            EventJson = context.EventJson,
        };
        eventCache[context.TaskId] = cached;
        ShowPopup(context.Identity, cached);

        if (!string.IsNullOrEmpty(context.BodyMeshUrl))
        {
            HTTPRequest bodyMeshRequest = new HTTPRequest(new Uri(context.BodyMeshUrl), HTTPMethods.Get, OnEventBodyMeshFinished);
            bodyMeshRequest.Tag = context;
            bodyMeshRequest.Send();
        }
        else if (!string.IsNullOrEmpty(context.SkeletonsUrl))
        {
            HTTPRequest skeletonRequest = new HTTPRequest(new Uri(context.SkeletonsUrl), HTTPMethods.Get, OnEventSkeletonsFinished);
            skeletonRequest.Tag = context;
            skeletonRequest.Send();
        }
    }

    private void OnEventBodyMeshFinished(HTTPRequest request, HTTPResponse response)
    {
        EventRequestContext context = request.Tag as EventRequestContext;
        if (!IsCurrentRequest(context) || activePopup == null || activeTaskId != context.TaskId)
        {
            return;
        }

        if (response == null || !response.IsSuccess)
        {
            Debug.LogWarning("[ModelEvent] Event body mesh request failed.");
            RequestSkeletonFallback(context);
            return;
        }

        JObject bodyMesh = JsonConvert.DeserializeObject(response.DataAsText) as JObject;
        JArray people = bodyMesh != null ? bodyMesh["people"] as JArray : null;
        if (people == null || people.Count == 0)
        {
            RequestSkeletonFallback(context);
            return;
        }

        if (eventCache.TryGetValue(context.TaskId, out CachedEventData cached) && cached != null)
        {
            cached.BodyMesh = bodyMesh;
            cached.Skeletons = null;
            DrawBodyMesh(activePopup.transform, cached);
        }
    }

    private void OnEventSkeletonsFinished(HTTPRequest request, HTTPResponse response)
    {
        EventRequestContext context = request.Tag as EventRequestContext;
        if (!IsCurrentRequest(context) || activePopup == null || activeTaskId != context.TaskId)
        {
            return;
        }

        if (response == null || !response.IsSuccess)
        {
            Debug.LogWarning("[ModelEvent] Event skeleton request failed.");
            return;
        }

        JObject root = JsonConvert.DeserializeObject(response.DataAsText) as JObject;
        JArray skeletons = root != null ? root["skeletons"] as JArray : null;
        if (skeletons == null)
        {
            return;
        }

        if (eventCache.TryGetValue(context.TaskId, out CachedEventData cached) && cached != null)
        {
            cached.Skeletons = skeletons;
            if (cached.BodyMesh == null)
            {
                DrawSkeletons(activePopup.transform, cached);
            }
        }
    }

    private void OnDeleteEventFinished(HTTPRequest request, HTTPResponse response)
    {
        string taskId = request.Tag as string ?? "";
        if (response == null || !response.IsSuccess)
        {
            string statusCode = response != null ? response.StatusCode.ToString(CultureInfo.InvariantCulture) : "no_response";
            Debug.LogWarning("[ModelEvent] Delete event failed: task_id=" + taskId + ", status=" + statusCode);
            return;
        }

        Debug.Log("[ModelEvent] Deleted server event for task_id=" + taskId);
    }

    private void ShowPopup(RuntimeModelEventIdentity identity, CachedEventData cached)
    {
        if (identity == null || cached == null || cached.Texture == null)
        {
            return;
        }

        Camera cam = ResolveCamera();
        if (cam == null)
        {
            ShowFrontMessage("model_event_ERR_no_camera");
            return;
        }

        identity.TryGetWorldBounds(out Bounds bounds);
        Vector3 toCamera = cam.transform.position - bounds.center;
        if (toCamera.sqrMagnitude < 0.0001f)
        {
            toCamera = -cam.transform.forward;
        }
        Vector3 popupPosition = bounds.center + toCamera.normalized * Mathf.Max(0.01f, popupDistanceFromBoxMeters);
        Quaternion popupRotation = Quaternion.LookRotation(cam.transform.position - popupPosition, cam.transform.up);

        GameObject root = new GameObject("ModelEventPopup_" + identity.TaskId);
        root.transform.SetPositionAndRotation(popupPosition, popupRotation);

        float width = Mathf.Max(0.05f, panelWidthMeters);
        float height = width * ((float)cached.Texture.height / Mathf.Max(1, cached.Texture.width));

        GameObject imageQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        imageQuad.name = "EventImage";
        imageQuad.transform.SetParent(root.transform, false);
        imageQuad.transform.localScale = new Vector3(width, height, 1f);
        Collider imageCollider = imageQuad.GetComponent<Collider>();
        if (imageCollider != null)
        {
            Destroy(imageCollider);
        }

        Renderer imageRenderer = imageQuad.GetComponent<Renderer>();
        if (imageRenderer != null)
        {
            Material imageMaterial = ResolveImageMaterial();
            imageMaterial.mainTexture = cached.Texture;
            imageMaterial.color = imageColor;
            imageRenderer.sharedMaterial = imageMaterial;
        }

        cached.PanelWidth = width;
        cached.PanelHeight = height;
        activePopup = root;
        activeTaskId = !string.IsNullOrEmpty(identity.TaskId) ? identity.TaskId : identity.ModelKey;

        if (cached.BodyMesh != null)
        {
            DrawBodyMesh(root.transform, cached);
        }
        else
        {
            DrawSkeletons(root.transform, cached);
        }
        ShowFrontMessage("model_event_show");
    }

    private void DrawBodyMesh(Transform popupRoot, CachedEventData cached)
    {
        if (popupRoot == null || cached == null || cached.Texture == null || cached.BodyMesh == null)
        {
            return;
        }

        DestroyChild(popupRoot, "SkeletonOverlay");
        DestroyChild(popupRoot, "BodyMeshOverlay");

        JArray people = cached.BodyMesh["people"] as JArray;
        if (people == null || people.Count == 0)
        {
            return;
        }

        GameObject overlay = new GameObject("BodyMeshOverlay");
        overlay.transform.SetParent(popupRoot, false);
        overlay.transform.localPosition = Vector3.zero;
        overlay.transform.localRotation = Quaternion.identity;
        overlay.transform.localScale = Vector3.one;

        int personIndex = 0;
        foreach (JObject person in people.Children<JObject>())
        {
            Mesh mesh = BuildBodyMesh(person, cached.PanelWidth, cached.PanelHeight);
            if (mesh == null)
            {
                continue;
            }

            GameObject meshObject = new GameObject("body_mesh_" + personIndex.ToString(CultureInfo.InvariantCulture));
            meshObject.transform.SetParent(overlay.transform, false);
            meshObject.transform.localPosition = Vector3.zero;
            meshObject.transform.localRotation = Quaternion.identity;
            meshObject.transform.localScale = Vector3.one;

            MeshFilter filter = meshObject.AddComponent<MeshFilter>();
            filter.sharedMesh = mesh;

            MeshRenderer renderer = meshObject.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = ResolveBodyMeshMaterial();
            personIndex++;
        }
    }

    private Mesh BuildBodyMesh(JObject person, float width, float height)
    {
        JArray vertexArray = person["vertices"] as JArray;
        JArray triangleArray = person["triangles"] as JArray;
        if (vertexArray == null || triangleArray == null || vertexArray.Count < 3 || triangleArray.Count < 3)
        {
            return null;
        }

        List<Vector3> vertices = new List<Vector3>(vertexArray.Count);
        foreach (JArray rawVertex in vertexArray.Children<JArray>())
        {
            if (rawVertex.Count < 2)
            {
                continue;
            }

            float x = rawVertex[0].Value<float>() * width;
            float y = rawVertex[1].Value<float>() * height;
            float z = rawVertex.Count >= 3 ? rawVertex[2].Value<float>() : 0.006f;
            vertices.Add(new Vector3(x, y, z));
        }

        if (vertices.Count < 3)
        {
            return null;
        }

        List<int> triangles = new List<int>(triangleArray.Count * 2);
        for (int i = 0; i + 2 < triangleArray.Count; i += 3)
        {
            int a = triangleArray[i].Value<int>();
            int b = triangleArray[i + 1].Value<int>();
            int c = triangleArray[i + 2].Value<int>();
            if (a < 0 || b < 0 || c < 0 || a >= vertices.Count || b >= vertices.Count || c >= vertices.Count)
            {
                continue;
            }

            triangles.Add(a);
            triangles.Add(b);
            triangles.Add(c);
            triangles.Add(c);
            triangles.Add(b);
            triangles.Add(a);
        }

        if (triangles.Count < 3)
        {
            return null;
        }

        Mesh mesh = new Mesh();
        if (vertices.Count > 65535)
        {
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        }
        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    private void RequestSkeletonFallback(EventRequestContext context)
    {
        if (context == null || string.IsNullOrEmpty(context.SkeletonsUrl))
        {
            return;
        }

        HTTPRequest skeletonRequest = new HTTPRequest(new Uri(context.SkeletonsUrl), HTTPMethods.Get, OnEventSkeletonsFinished);
        skeletonRequest.Tag = context;
        skeletonRequest.Send();
    }

    private void DrawSkeletons(Transform popupRoot, CachedEventData cached)
    {
        if (popupRoot == null || cached == null || cached.Texture == null || cached.Skeletons == null)
        {
            return;
        }

        DestroyChild(popupRoot, "SkeletonOverlay");

        GameObject overlay = new GameObject("SkeletonOverlay");
        overlay.transform.SetParent(popupRoot, false);
        overlay.transform.localPosition = Vector3.forward * 0.002f;
        overlay.transform.localRotation = Quaternion.identity;
        overlay.transform.localScale = Vector3.one;

        string contactPeopleId = cached.EventJson != null ? cached.EventJson["hand_contact"]?["people_id"]?.ToString() : "";
        string contactHand = cached.EventJson != null ? cached.EventJson["hand_contact"]?["hand"]?.ToString() : "";

        foreach (JObject skeleton in cached.Skeletons.Children<JObject>())
        {
            string peopleId = skeleton["people_id"]?.ToString() ?? "";
            if (!string.IsNullOrEmpty(contactPeopleId) && peopleId != contactPeopleId)
            {
                continue;
            }

            Dictionary<string, Vector3> joints = BuildJointMap(skeleton, cached.Texture, cached.PanelWidth, cached.PanelHeight);
            for (int i = 0; i < SkeletonPairs.GetLength(0); i++)
            {
                string a = SkeletonPairs[i, 0];
                string b = SkeletonPairs[i, 1];
                if (joints.TryGetValue(a, out Vector3 pa) && joints.TryGetValue(b, out Vector3 pb))
                {
                    CreateLine(overlay.transform, pa, pb, runtimeSkeletonMaterial, skeletonLineColor, "bone_" + a + "_" + b);
                }
            }

            foreach (KeyValuePair<string, Vector3> item in joints)
            {
                bool isWrist = item.Key == "left_wrist" || item.Key == "right_wrist";
                bool isContact = isWrist
                    && !string.IsNullOrEmpty(contactPeopleId)
                    && contactPeopleId == peopleId
                    && (string.IsNullOrEmpty(contactHand) || contactHand == item.Key);
                CreateJointPoint(
                    overlay.transform,
                    item.Value,
                    isWrist ? runtimeWristMaterial : runtimeSkeletonMaterial,
                    isContact ? jointPointSizeMeters * 1.6f : jointPointSizeMeters,
                    "joint_" + item.Key
                );
            }
        }
    }

    private Dictionary<string, Vector3> BuildJointMap(JObject skeleton, Texture2D texture, float width, float height)
    {
        Dictionary<string, Vector3> joints = new Dictionary<string, Vector3>();
        JArray jointArray = skeleton["joints"] as JArray;
        if (jointArray == null)
        {
            return joints;
        }

        foreach (JObject joint in jointArray.Children<JObject>())
        {
            string name = joint["body_part_name"]?.ToString() ?? "";
            if (string.IsNullOrEmpty(name))
            {
                continue;
            }

            float score = joint["score"] != null ? joint["score"].Value<float>() : 1f;
            if (score < minJointScore)
            {
                continue;
            }

            JArray pixel = joint["pixel_xy"] as JArray;
            if (pixel == null || pixel.Count < 2)
            {
                continue;
            }

            float px = pixel[0].Value<float>();
            float py = pixel[1].Value<float>();
            if (float.IsNaN(px) || float.IsNaN(py))
            {
                continue;
            }

            float x = ((px / Mathf.Max(1, texture.width)) - 0.5f) * width;
            float y = (0.5f - (py / Mathf.Max(1, texture.height))) * height;
            joints[name] = new Vector3(x, y, 0.004f);
        }

        return joints;
    }

    private void CreateLine(Transform parent, Vector3 a, Vector3 b, Material material, Color color, string name)
    {
        GameObject lineObject = new GameObject(name);
        lineObject.transform.SetParent(parent, false);
        LineRenderer line = lineObject.AddComponent<LineRenderer>();
        line.useWorldSpace = false;
        line.positionCount = 2;
        line.SetPosition(0, a);
        line.SetPosition(1, b);
        line.startWidth = skeletonLineWidth;
        line.endWidth = skeletonLineWidth;
        line.startColor = color;
        line.endColor = color;
        line.material = material;
    }

    private void CreateJointPoint(Transform parent, Vector3 position, Material material, float size, string name)
    {
        GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        point.name = name;
        point.transform.SetParent(parent, false);
        point.transform.localPosition = position;
        point.transform.localScale = Vector3.one * Mathf.Max(0.002f, size);
        Renderer renderer = point.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material = material;
        }
        Collider collider = point.GetComponent<Collider>();
        if (collider != null)
        {
            Destroy(collider);
        }
    }

    private void DestroyChild(Transform parent, string childName)
    {
        Transform existing = parent != null ? parent.Find(childName) : null;
        if (existing != null)
        {
            Destroy(existing.gameObject);
        }
    }

    private bool IsCurrentRequest(EventRequestContext context)
    {
        return context != null
            && context.Generation == requestGeneration
            && context.Identity != null;
    }

    private Camera ResolveCamera()
    {
        if (targetCamera != null)
        {
            return targetCamera;
        }

        targetCamera = Camera.main;
        return targetCamera;
    }

    private string NormalizeServerBaseUrl()
    {
        string value = string.IsNullOrEmpty(serverBaseUrl) ? "http://10.40.1.122:7355" : serverBaseUrl.Trim();
        return value.TrimEnd('/');
    }

    private void ClosePopup()
    {
        if (activePopup != null)
        {
            Destroy(activePopup);
            activePopup = null;
        }
        activeTaskId = "";
    }

    private void EnsureMaterials()
    {
        ResolveImageMaterial();
        if (runtimeSkeletonMaterial == null)
        {
            runtimeSkeletonMaterial = CreateColorMaterial(skeletonLineColor);
        }
        if (runtimeWristMaterial == null)
        {
            runtimeWristMaterial = CreateColorMaterial(wristColor);
        }
        if (runtimeBodyMeshMaterial == null)
        {
            runtimeBodyMeshMaterial = CreateTransparentMeshMaterial(bodyMeshColor);
        }
    }

    private Material ResolveImageMaterial()
    {
        if (runtimeImageMaterial != null)
        {
            return runtimeImageMaterial;
        }

        if (imageMaterialTemplate != null)
        {
            runtimeImageMaterial = new Material(imageMaterialTemplate);
            return runtimeImageMaterial;
        }

        Shader shader = Shader.Find("Unlit/Texture");
        if (shader == null)
        {
            shader = Shader.Find("Standard");
        }
        runtimeImageMaterial = new Material(shader);
        return runtimeImageMaterial;
    }

    private Material CreateColorMaterial(Color color)
    {
        if (skeletonLineMaterial != null)
        {
            Material material = new Material(skeletonLineMaterial);
            material.color = color;
            return material;
        }

        Shader shader = Shader.Find("Sprites/Default");
        if (shader == null)
        {
            shader = Shader.Find("Unlit/Color");
        }
        if (shader == null)
        {
            shader = Shader.Find("Standard");
        }

        Material created = new Material(shader);
        created.color = color;
        return created;
    }

    private Material ResolveBodyMeshMaterial()
    {
        if (runtimeBodyMeshMaterial != null)
        {
            return runtimeBodyMeshMaterial;
        }

        runtimeBodyMeshMaterial = CreateTransparentMeshMaterial(bodyMeshColor);
        return runtimeBodyMeshMaterial;
    }

    private Material CreateTransparentMeshMaterial(Color color)
    {
        Material material;
        if (bodyMeshMaterialTemplate != null)
        {
            material = new Material(bodyMeshMaterialTemplate);
        }
        else
        {
            Shader shader = Shader.Find("Unlit/Color");
            if (shader == null)
            {
                shader = Shader.Find("Standard");
            }
            material = new Material(shader);
        }

        material.color = color;
        if (material.HasProperty("_Color"))
        {
            material.SetColor("_Color", color);
        }
        if (material.HasProperty("_Cull"))
        {
            material.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
        }
        if (material.HasProperty("_Mode"))
        {
            material.SetFloat("_Mode", 3f);
        }
        if (material.HasProperty("_SrcBlend"))
        {
            material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        }
        if (material.HasProperty("_DstBlend"))
        {
            material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        }
        if (material.HasProperty("_ZWrite"))
        {
            material.SetInt("_ZWrite", 0);
        }
        material.DisableKeyword("_ALPHATEST_ON");
        material.EnableKeyword("_ALPHABLEND_ON");
        material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        return material;
    }

    private void DestroyMaterial(Material material)
    {
        if (material != null)
        {
            Destroy(material);
        }
    }

    private void ShowFrontMessage(string message)
    {
        if (Game_M.initialize != null)
        {
            Game_M.initialize.XianShiForSeconds(message);
        }
    }

    private class EventRequestContext
    {
        public RuntimeModelEventIdentity Identity;
        public string TaskId;
        public int Generation;
        public JObject EventJson;
        public string SkeletonsUrl;
        public string BodyMeshUrl;
    }

    private class CachedEventData
    {
        public Texture2D Texture;
        public JObject EventJson;
        public JArray Skeletons;
        public JObject BodyMesh;
        public float PanelWidth;
        public float PanelHeight;
    }
}
