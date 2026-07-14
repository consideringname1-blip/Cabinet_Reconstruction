using BestHTTP;
using Newtonsoft.Json.Linq;
using System;
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
    private RuntimeModelManager runtimeModelManager;
    private long evidenceGeneration;

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
            if (item != null && item.ImageObject != null)
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
    }

    public bool ShowHistoryEvidence(
        string displayObjectId,
        string eventUid,
        JObject evidence,
        string coordinateEpoch)
    {
        if (string.IsNullOrEmpty(displayObjectId)
            || string.IsNullOrEmpty(eventUid)
            || string.IsNullOrEmpty(coordinateEpoch)
            || evidence == null
            || !HasExactKeys(evidence, "scene_image_url", "skeleton"))
        {
            Debug.LogWarning("[ObjectEvidence] Reject incomplete strict-v2 evidence.");
            return false;
        }

        if (!TryValidateHistoryEvidence(
                evidence,
                out Uri imageUri,
                out RuntimeSkeletonData skeleton))
        {
            Debug.LogWarning("[ObjectEvidence] Reject invalid strict-v2 evidence.");
            return false;
        }
        string imageUrl = imageUri.AbsoluteUri;

        HideEvidenceForModel(displayObjectId);
        EnsureRoot();
        EvidenceItem item = new EvidenceItem
        {
            DisplayObjectId = displayObjectId,
            EventUid = eventUid,
            CoordinateEpoch = coordinateEpoch,
            ImageUrl = imageUrl,
            Generation = ++evidenceGeneration,
        };
        evidenceByDisplayObjectId[displayObjectId] = item;

        item.SkeletonObject = new GameObject(
            "ObjectEvidenceSkeleton_" + displayObjectId);
        item.SkeletonObject.transform.SetParent(evidenceRoot.transform, false);
        item.SkeletonDisplay =
            item.SkeletonObject.AddComponent<RuntimeSkeletonDisplay>();
        item.SkeletonDisplay.Configure(skeleton);

        EvidenceImageRequestContext context = new EvidenceImageRequestContext
        {
            DisplayObjectId = displayObjectId,
            EventUid = eventUid,
            Url = imageUrl,
            Generation = item.Generation,
        };
        HTTPRequest request = new HTTPRequest(
            imageUri,
            HTTPMethods.Get,
            OnEvidenceImageDownloaded);
        request.Tag = context;
        item.ImageRequest = request;
        request.Send();
        return true;
    }

    public static bool ValidateHistoryEvidence(JObject evidence)
    {
        return TryValidateHistoryEvidence(
            evidence,
            out Uri ignoredUri,
            out RuntimeSkeletonData ignoredSkeleton);
    }

    public bool HideEvidenceForModel(string displayObjectId)
    {
        if (string.IsNullOrEmpty(displayObjectId)
            || !evidenceByDisplayObjectId.TryGetValue(
                displayObjectId,
                out EvidenceItem item))
        {
            return false;
        }
        evidenceByDisplayObjectId.Remove(displayObjectId);
        ReleaseItemVisuals(item);
        return true;
    }

    public int HideAll()
    {
        int hidden = evidenceByDisplayObjectId.Count;
        foreach (EvidenceItem item in
            new List<EvidenceItem>(evidenceByDisplayObjectId.Values))
        {
            ReleaseItemVisuals(item);
        }
        evidenceByDisplayObjectId.Clear();
        return hidden;
    }

    public void Clear()
    {
        HideAll();
        if (evidenceRoot != null)
        {
            Destroy(evidenceRoot);
            evidenceRoot = null;
        }
    }

    private void OnEvidenceImageDownloaded(
        HTTPRequest request,
        HTTPResponse response)
    {
        EvidenceImageRequestContext context = request != null
            ? request.Tag as EvidenceImageRequestContext
            : null;
        if (context == null
            || !evidenceByDisplayObjectId.TryGetValue(
                context.DisplayObjectId,
                out EvidenceItem item)
            || item == null
            || item.ImageRequest != request
            || item.Generation != context.Generation
            || item.EventUid != context.EventUid
            || item.ImageUrl != context.Url)
        {
            return;
        }

        item.ImageRequest = null;
        if (response == null
            || !response.IsSuccess
            || response.Data == null
            || response.Data.Length == 0)
        {
            ShowFrontMessage("object_evidence_ERR_rgb");
            return;
        }

        Texture2D texture = new Texture2D(
            2,
            2,
            TextureFormat.RGBA32,
            false);
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
            if (shader == null)
            {
                shader = Shader.Find("Standard");
            }
            if (shader == null)
            {
                shader = Shader.Find("Hidden/InternalErrorShader");
            }
            item.ImageMaterial = new Material(shader);
            item.ImageMaterial.mainTexture = texture;
            renderer.sharedMaterial = item.ImageMaterial;
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
            : Vector3.Lerp(
                item.ImageObject.transform.position,
                target,
                Time.deltaTime * ImageLerpSpeed);

        Camera camera = Camera.main;
        if (camera != null)
        {
            Vector3 toCamera =
                item.ImageObject.transform.position - camera.transform.position;
            if (toCamera.sqrMagnitude > 0.0001f)
            {
                item.ImageObject.transform.rotation =
                    Quaternion.LookRotation(toCamera.normalized, Vector3.up);
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
            && TryGetRendererBounds(record.RootGameObject, out Bounds modelBounds))
        {
            return new Vector3(
                modelBounds.center.x,
                modelBounds.max.y,
                modelBounds.center.z);
        }
        if (item.SkeletonObject != null
            && TryGetRendererBounds(
                item.SkeletonObject,
                out Bounds skeletonBounds))
        {
            return new Vector3(
                skeletonBounds.center.x,
                skeletonBounds.max.y,
                skeletonBounds.center.z);
        }
        Camera camera = Camera.main;
        return camera != null
            ? camera.transform.position + camera.transform.forward
            : Vector3.zero;
    }

    private void ReleaseItemVisuals(EvidenceItem item)
    {
        if (item == null)
        {
            return;
        }
        if (item.ImageRequest != null)
        {
            item.ImageRequest.Abort();
            item.ImageRequest = null;
        }
        ReleaseImage(item);
        if (item.SkeletonObject != null)
        {
            Destroy(item.SkeletonObject);
            item.SkeletonObject = null;
            item.SkeletonDisplay = null;
        }
    }

    private void ReleaseImage(EvidenceItem item)
    {
        if (item == null)
        {
            return;
        }
        if (item.ImageObject != null)
        {
            Destroy(item.ImageObject);
            item.ImageObject = null;
        }
        if (item.ImageMaterial != null)
        {
            Destroy(item.ImageMaterial);
            item.ImageMaterial = null;
        }
        if (item.ImageTexture != null)
        {
            Destroy(item.ImageTexture);
            item.ImageTexture = null;
        }
    }

    private static bool TryParseSkeleton(
        JObject payload,
        out RuntimeSkeletonData skeleton)
    {
        skeleton = null;
        if (payload == null
            || !HasExactKeys(payload, "people_id", "joints"))
        {
            return false;
        }
        string peopleId = ReadRequiredString(payload, "people_id");
        JArray joints = payload["joints"] as JArray;
        if (string.IsNullOrEmpty(peopleId)
            || joints == null
            || joints.Count == 0)
        {
            return false;
        }

        RuntimeSkeletonData parsed = new RuntimeSkeletonData
        {
            PeopleId = peopleId,
        };
        HashSet<string> names = new HashSet<string>(StringComparer.Ordinal);
        foreach (JToken token in joints)
        {
            JObject joint = token as JObject;
            if (joint == null
                || !HasExactKeys(
                    joint,
                    "name",
                    "position",
                    "score",
                    "valid"))
            {
                return false;
            }
            string name = ReadRequiredString(joint, "name");
            JToken scoreToken = joint["score"];
            JToken validToken = joint["valid"];
            float score = IsJsonNumber(scoreToken)
                ? scoreToken.Value<float>()
                : float.NaN;
            if (string.IsNullOrEmpty(name)
                || !names.Add(name)
                || !TryReadVector3(joint["position"], out Vector3 position)
                || !IsJsonNumber(scoreToken)
                || float.IsNaN(score)
                || float.IsInfinity(score)
                || validToken == null
                || validToken.Type != JTokenType.Boolean)
            {
                return false;
            }
            parsed.Joints.Add(new RuntimeSkeletonJointData
            {
                Name = name,
                PositionWorld = position,
                Score = score,
                Valid = validToken.Value<bool>(),
            });
        }
        skeleton = parsed;
        return true;
    }

    private static bool TryValidateHistoryEvidence(
        JObject evidence,
        out Uri imageUri,
        out RuntimeSkeletonData skeleton)
    {
        imageUri = null;
        skeleton = null;
        if (evidence == null
            || !HasExactKeys(evidence, "scene_image_url", "skeleton"))
        {
            return false;
        }
        string imageUrl = ReadRequiredString(evidence, "scene_image_url");
        return Uri.TryCreate(imageUrl, UriKind.Absolute, out imageUri)
            && evidence["skeleton"] is JObject skeletonObject
            && TryParseSkeleton(skeletonObject, out skeleton);
    }

    private static bool TryReadVector3(JToken token, out Vector3 value)
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

    private static bool TryGetRendererBounds(
        GameObject root,
        out Bounds bounds)
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
        public string EventUid = "";
        public string CoordinateEpoch = "";
        public string ImageUrl = "";
        public long Generation;
        public HTTPRequest ImageRequest;
        public Texture2D ImageTexture;
        public Material ImageMaterial;
        public GameObject ImageObject;
        public GameObject SkeletonObject;
        public RuntimeSkeletonDisplay SkeletonDisplay;
    }

    private sealed class EvidenceImageRequestContext
    {
        public string DisplayObjectId = "";
        public string EventUid = "";
        public string Url = "";
        public long Generation;
    }
}
