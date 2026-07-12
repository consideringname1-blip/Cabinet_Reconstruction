using BestHTTP;
using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class ObjectEvidenceDisplay : MonoBehaviour
{
    private const string RootName = "ObjectEvidenceDisplayRoot";
    private const float ImageVerticalOffsetMeters = 0.30f;
    private const float ImageMaxWidthMeters = 0.43f;
    private const float ImageMaxHeightMeters = 0.24f;
    private const float ImageLerpSpeed = 8.0f;

    private static ObjectEvidenceDisplay _instance;
    private readonly Dictionary<string, EvidenceItem> evidenceByDisplayObjectId =
        new Dictionary<string, EvidenceItem>(StringComparer.Ordinal);
    private GameObject evidenceRoot;
    private Material evidenceBodyMaterial;
    private RuntimeModelManager runtimeModelManager;

    public static ObjectEvidenceDisplay Instance
    {
        get
        {
            if (_instance != null)
            {
                return _instance;
            }

            _instance = FindObjectOfType<ObjectEvidenceDisplay>();
            if (_instance == null)
            {
                GameObject displayObject = new GameObject("ObjectEvidenceDisplay");
                _instance = displayObject.AddComponent<ObjectEvidenceDisplay>();
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

    private void Update()
    {
        foreach (EvidenceItem item in evidenceByDisplayObjectId.Values)
        {
            if (item != null && item.ImageObject != null && item.ImageObject.activeSelf)
            {
                UpdateImagePlacement(item, false);
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
        if (evidenceBodyMaterial != null)
        {
            Destroy(evidenceBodyMaterial);
            evidenceBodyMaterial = null;
        }
    }

    public void Clear()
    {
        foreach (EvidenceItem item in new List<EvidenceItem>(evidenceByDisplayObjectId.Values))
        {
            ReleaseItemVisuals(item);
        }
        evidenceByDisplayObjectId.Clear();
        if (evidenceRoot != null)
        {
            Destroy(evidenceRoot);
            evidenceRoot = null;
        }
    }

    public bool RegisterEvidence(JObject evidence)
    {
        if (evidence == null)
        {
            return false;
        }

        string displayObjectId = ReadRequiredString(evidence, "display_object_id");
        string taskId = ReadRequiredString(evidence, "task_id");
        string imageUrl = ReadRequiredString(evidence, "image_url");
        JObject bodyModel = evidence["body_model"] as JObject;
        long bodyRevision = ReadRequiredLong(evidence, "body_revision");
        if (!HasExactKeys(
                evidence,
                "task_id",
                "display_object_id",
                "body_revision",
                "image_url",
                "body_model")
            || string.IsNullOrEmpty(displayObjectId)
            || string.IsNullOrEmpty(taskId)
            || string.IsNullOrEmpty(imageUrl)
            || bodyModel == null
            || bodyRevision <= 0)
        {
            Debug.LogWarning("[ObjectEvidence] Reject incomplete canonical body_evidence payload.");
            return false;
        }

        string bodyModelKey = ReadRequiredString(bodyModel, "model_key");
        string bodyFbxUrl = ReadRequiredString(bodyModel, "fbx_url");
        if (!HasExactKeys(
                bodyModel,
                "model_key",
                "task_id",
                "display_object_id",
                "model_revision",
                "fbx_url",
                "pose",
                "coordinate_space")
            || bodyModelKey != "body:" + displayObjectId
            || ReadRequiredString(bodyModel, "task_id") != taskId
            || ReadRequiredString(bodyModel, "display_object_id") != displayObjectId
            || ReadRequiredLong(bodyModel, "model_revision") != bodyRevision
            || string.IsNullOrEmpty(bodyFbxUrl)
            || ReadRequiredString(bodyModel, "coordinate_space") != "hololens_current_local"
            || !(bodyModel["pose"] is JObject))
        {
            Debug.LogWarning("[ObjectEvidence] Reject invalid canonical body_model.");
            return false;
        }

        EvidenceItem item;
        if (!evidenceByDisplayObjectId.TryGetValue(displayObjectId, out item) || item == null)
        {
            item = new EvidenceItem
            {
                DisplayObjectId = displayObjectId,
                BodyModelKey = bodyModelKey,
            };
        }

        bool revisionChanged = item.BodyRevision != bodyRevision;
        bool imageChanged = !string.Equals(item.ImageUrl, imageUrl, StringComparison.Ordinal);
        bool bodyChanged = !string.Equals(item.BodyFbxUrl, bodyFbxUrl, StringComparison.Ordinal);
        if (revisionChanged || imageChanged)
        {
            ReleaseImage(item);
        }
        if (revisionChanged || bodyChanged)
        {
            item.BodyDownloadQueued = false;
            if (item.BodyVisibilityCoroutine != null)
            {
                StopCoroutine(item.BodyVisibilityCoroutine);
                item.BodyVisibilityCoroutine = null;
            }
            HideLoadedBody(item);
        }

        item.BodyRevision = bodyRevision;
        item.ImageUrl = imageUrl;
        item.BodyFbxUrl = bodyFbxUrl;
        item.BodyModel = (JObject)bodyModel.DeepClone();
        evidenceByDisplayObjectId[displayObjectId] = item;
        return true;
    }

    public bool ToggleEvidenceForModel(string displayObjectId)
    {
        if (string.IsNullOrEmpty(displayObjectId)
            || !evidenceByDisplayObjectId.TryGetValue(displayObjectId, out EvidenceItem item)
            || item == null)
        {
            return false;
        }

        if (IsEvidenceVisibleOrPending(item))
        {
            HideEvidence(item);
            return true;
        }

        item.VisibleRequested = true;
        EnsureRoot();
        bool available = EnsureImage(item) | EnsureBodyMesh(item);
        if (!available)
        {
            ShowFrontMessage("object_evidence_no_visual_artifact");
        }
        return true;
    }

    private bool IsEvidenceVisibleOrPending(EvidenceItem item)
    {
        if (item.ImageRequestInFlight || item.BodyDownloadQueued)
        {
            return true;
        }
        if (item.ImageObject != null && item.ImageObject.activeSelf)
        {
            return true;
        }
        ResolveRuntimeModelManager();
        return runtimeModelManager != null
            && runtimeModelManager.TryGetLoadedRecord(item.BodyModelKey, out RuntimeModelRecord bodyRecord)
            && bodyRecord != null
            && bodyRecord.RootGameObject != null
            && bodyRecord.RootGameObject.activeSelf;
    }

    private void HideEvidence(EvidenceItem item)
    {
        item.VisibleRequested = false;
        if (item.ImageObject != null)
        {
            item.ImageObject.SetActive(false);
        }
        HideLoadedBody(item);
    }

    private bool EnsureImage(EvidenceItem item)
    {
        if (item.ImageObject != null)
        {
            item.ImageObject.SetActive(item.VisibleRequested);
            return true;
        }
        if (item.ImageRequestInFlight)
        {
            return true;
        }
        if (string.IsNullOrEmpty(item.ImageUrl))
        {
            return false;
        }

        item.ImageRequestInFlight = true;
        EvidenceImageRequestContext context = new EvidenceImageRequestContext
        {
            DisplayObjectId = item.DisplayObjectId,
            BodyRevision = item.BodyRevision,
            Url = item.ImageUrl,
        };
        var request = new HTTPRequest(new Uri(item.ImageUrl), HTTPMethods.Get, OnEvidenceImageDownloaded);
        request.Tag = context;
        request.Send();
        return true;
    }

    private void OnEvidenceImageDownloaded(HTTPRequest request, HTTPResponse response)
    {
        EvidenceImageRequestContext context = request != null
            ? request.Tag as EvidenceImageRequestContext
            : null;
        if (context == null
            || !evidenceByDisplayObjectId.TryGetValue(context.DisplayObjectId, out EvidenceItem item)
            || item == null
            || item.BodyRevision != context.BodyRevision
            || !string.Equals(item.ImageUrl, context.Url, StringComparison.Ordinal))
        {
            return;
        }

        item.ImageRequestInFlight = false;
        if (response == null || !response.IsSuccess || response.Data == null || response.Data.Length == 0)
        {
            ShowFrontMessage("object_evidence_ERR_rgb");
            return;
        }

        Texture2D texture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        if (!texture.LoadImage(response.Data))
        {
            Destroy(texture);
            ShowFrontMessage("object_evidence_ERR_rgb");
            return;
        }

        ReleaseImage(item);
        item.ImageTexture = texture;
        item.ImageObject = CreateImageQuad(item, texture);
        if (item.ImageObject != null)
        {
            item.ImageObject.SetActive(item.VisibleRequested);
            UpdateImagePlacement(item, true);
        }
    }

    private GameObject CreateImageQuad(EvidenceItem item, Texture2D texture)
    {
        EnsureRoot();
        if (evidenceRoot == null)
        {
            return null;
        }

        GameObject quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.name = "ObjectEvidenceRgb_" + item.DisplayObjectId;
        quad.transform.SetParent(evidenceRoot.transform, false);
        Collider collider = quad.GetComponent<Collider>();
        if (collider != null)
        {
            Destroy(collider);
        }

        float aspect = texture.height > 0
            ? Mathf.Max(0.001f, (float)texture.width / texture.height)
            : 1.0f;
        float width = ImageMaxWidthMeters;
        float height = width / aspect;
        if (height > ImageMaxHeightMeters)
        {
            height = ImageMaxHeightMeters;
            width = height * aspect;
        }
        quad.transform.localScale = new Vector3(width, height, 1.0f);

        Renderer renderer = quad.GetComponent<Renderer>();
        if (renderer != null)
        {
            Shader shader = Shader.Find("Unlit/Texture");
            renderer.material = new Material(shader != null ? shader : Shader.Find("Standard"));
            renderer.material.mainTexture = texture;
        }
        return quad;
    }

    private void UpdateImagePlacement(EvidenceItem item, bool force)
    {
        if (item == null || item.ImageObject == null)
        {
            return;
        }

        Vector3 anchor = ResolveEvidenceAnchor(item);
        Vector3 target = anchor + Vector3.up * ImageVerticalOffsetMeters;
        item.ImageObject.transform.position = force
            ? target
            : Vector3.Lerp(item.ImageObject.transform.position, target, Time.deltaTime * ImageLerpSpeed);

        Camera camera = Camera.main;
        if (camera != null)
        {
            Vector3 toCamera = item.ImageObject.transform.position - camera.transform.position;
            if (toCamera.sqrMagnitude > 0.0001f)
            {
                item.ImageObject.transform.rotation = Quaternion.LookRotation(toCamera.normalized, Vector3.up);
            }
        }
    }

    private Vector3 ResolveEvidenceAnchor(EvidenceItem item)
    {
        ResolveRuntimeModelManager();
        if (runtimeModelManager != null
            && runtimeModelManager.TryGetLoadedRecordByDisplayObjectId(
                item.DisplayObjectId,
                out RuntimeModelRecord record)
            && record != null
            && record.RootGameObject != null
            && TryGetRendererBounds(record.RootGameObject, out Bounds bounds))
        {
            return new Vector3(bounds.center.x, bounds.max.y, bounds.center.z);
        }
        return Vector3.zero;
    }

    private bool EnsureBodyMesh(EvidenceItem item)
    {
        if (string.IsNullOrEmpty(item.BodyFbxUrl)
            || item.BodyRevision <= 0
            || item.BodyModel == null)
        {
            return false;
        }
        if (item.BodyDownloadQueued)
        {
            return true;
        }

        ResolveRuntimeModelManager();
        if (runtimeModelManager != null
            && runtimeModelManager.TryGetLoadedRecord(item.BodyModelKey, out RuntimeModelRecord bodyRecord)
            && bodyRecord != null
            && bodyRecord.ModelRevision == item.BodyRevision
            && string.Equals(bodyRecord.FbxUrl, item.BodyFbxUrl, StringComparison.Ordinal)
            && bodyRecord.RootGameObject != null)
        {
            ApplyEvidenceBodyMaterial(bodyRecord.RootGameObject);
            bodyRecord.RootGameObject.SetActive(item.VisibleRequested);
            return true;
        }

        if (ShuJuQingQiu.initialize == null)
        {
            return false;
        }
        if (!ShuJuQingQiu.initialize.QueueRuntimeModelDownload(
                (JObject)item.BodyModel.DeepClone(),
                true))
        {
            return false;
        }

        item.BodyDownloadQueued = true;
        if (item.BodyVisibilityCoroutine != null)
        {
            StopCoroutine(item.BodyVisibilityCoroutine);
        }
        item.BodyVisibilityCoroutine = StartCoroutine(WaitForBodyMesh(item));
        return true;
    }

    private IEnumerator WaitForBodyMesh(EvidenceItem item)
    {
        long expectedRevision = item.BodyRevision;
        float timeoutAt = Time.time + 30f;
        while (item != null && item.BodyRevision == expectedRevision && Time.time < timeoutAt)
        {
            ResolveRuntimeModelManager();
            if (runtimeModelManager != null
                && runtimeModelManager.TryGetLoadedRecord(item.BodyModelKey, out RuntimeModelRecord bodyRecord)
                && bodyRecord != null
                && bodyRecord.ModelRevision == expectedRevision
                && bodyRecord.RootGameObject != null)
            {
                ApplyEvidenceBodyMaterial(bodyRecord.RootGameObject);
                bodyRecord.RootGameObject.SetActive(item.VisibleRequested);
                item.BodyDownloadQueued = false;
                item.BodyVisibilityCoroutine = null;
                yield break;
            }
            yield return null;
        }
        if (item != null && item.BodyRevision == expectedRevision)
        {
            item.BodyDownloadQueued = false;
            item.BodyVisibilityCoroutine = null;
        }
    }

    private void HideLoadedBody(EvidenceItem item)
    {
        ResolveRuntimeModelManager();
        if (runtimeModelManager != null
            && runtimeModelManager.TryGetLoadedRecord(item.BodyModelKey, out RuntimeModelRecord bodyRecord)
            && bodyRecord != null
            && bodyRecord.RootGameObject != null)
        {
            bodyRecord.RootGameObject.SetActive(false);
        }
    }

    private void ApplyEvidenceBodyMaterial(GameObject root)
    {
        if (root == null)
        {
            return;
        }
        if (evidenceBodyMaterial == null)
        {
            Shader shader = Shader.Find("Standard");
            evidenceBodyMaterial = new Material(shader);
            evidenceBodyMaterial.color = new Color(0.62f, 0.66f, 0.70f, 0.38f);
            if (evidenceBodyMaterial.HasProperty("_Mode"))
            {
                evidenceBodyMaterial.SetFloat("_Mode", 3f);
                evidenceBodyMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                evidenceBodyMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                evidenceBodyMaterial.SetInt("_ZWrite", 0);
                evidenceBodyMaterial.EnableKeyword("_ALPHABLEND_ON");
                evidenceBodyMaterial.renderQueue = 3000;
            }
        }
        foreach (Renderer renderer in root.GetComponentsInChildren<Renderer>(true))
        {
            if (renderer != null)
            {
                renderer.sharedMaterial = evidenceBodyMaterial;
            }
        }
    }

    private void ReleaseItemVisuals(EvidenceItem item)
    {
        if (item == null)
        {
            return;
        }
        if (item.BodyVisibilityCoroutine != null)
        {
            StopCoroutine(item.BodyVisibilityCoroutine);
            item.BodyVisibilityCoroutine = null;
        }
        HideLoadedBody(item);
        ReleaseImage(item);
    }

    private void ReleaseImage(EvidenceItem item)
    {
        item.ImageRequestInFlight = false;
        if (item.ImageObject != null)
        {
            Destroy(item.ImageObject);
            item.ImageObject = null;
        }
        if (item.ImageTexture != null)
        {
            Destroy(item.ImageTexture);
            item.ImageTexture = null;
        }
    }

    private static bool TryGetRendererBounds(GameObject root, out Bounds bounds)
    {
        bounds = new Bounds(Vector3.zero, Vector3.zero);
        bool initialized = false;
        if (root == null)
        {
            return false;
        }
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

    private static string ReadRequiredString(JObject payload, string key)
    {
        JToken token = payload != null ? payload[key] : null;
        return token != null && token.Type == JTokenType.String
            ? token.Value<string>().Trim()
            : "";
    }

    private static long ReadRequiredLong(JObject payload, string key)
    {
        JToken token = payload != null ? payload[key] : null;
        return token != null && token.Type == JTokenType.Integer ? token.Value<long>() : -1;
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

    private void ResolveRuntimeModelManager()
    {
        if (runtimeModelManager == null)
        {
            runtimeModelManager = RuntimeModelManager.Instance;
        }
    }

    private void EnsureRoot()
    {
        if (evidenceRoot == null)
        {
            evidenceRoot = new GameObject(RootName);
        }
    }

    private static void ShowFrontMessage(string message)
    {
        if (Game_M.initialize != null)
        {
            Game_M.initialize.XianShi(message);
        }
    }

    private sealed class EvidenceItem
    {
        public string DisplayObjectId = "";
        public long BodyRevision = -1;
        public string ImageUrl = "";
        public string BodyFbxUrl = "";
        public string BodyModelKey = "";
        public JObject BodyModel;
        public bool ImageRequestInFlight;
        public bool BodyDownloadQueued;
        public bool VisibleRequested;
        public Coroutine BodyVisibilityCoroutine;
        public Texture2D ImageTexture;
        public GameObject ImageObject;
    }

    private sealed class EvidenceImageRequestContext
    {
        public string DisplayObjectId = "";
        public long BodyRevision = -1;
        public string Url = "";
    }
}
