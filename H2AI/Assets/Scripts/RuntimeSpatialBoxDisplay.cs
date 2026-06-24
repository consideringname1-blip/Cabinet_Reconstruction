using System;
using System.Globalization;
using UnityEngine;

[DisallowMultipleComponent]
public class RuntimeSpatialBoxDisplay : MonoBehaviour
{
    private const float PanelOffsetMeters = 0.20f;
    private const float FaceSwitchHoldSeconds = 1.0f;
    private const float PanelLerpSpeed = 8.0f;

    private static readonly int[,] EdgePairs =
    {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
        { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
        { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
    };

    private RuntimeSpatialBoxData box;
    private GameObject fillObject;
    private Transform panelRoot;
    private TextMesh panelText;
    private Material fillMaterial;
    private Material lineMaterial;
    private Material panelMaterial;
    private Vector3[] corners = new Vector3[8];
    private Vector3 activeFaceNormal = Vector3.forward;
    private Vector3 pendingFaceNormal = Vector3.forward;
    private float pendingFaceSince;
    private bool faceInitialized;

    public void Configure(RuntimeSpatialBoxData spatialBox, string message)
    {
        box = spatialBox;
        Rebuild();
        UpdateMessage(message);
    }

    public void UpdateMessage(string message)
    {
        if (panelText != null)
        {
            panelText.text = string.IsNullOrEmpty(message) ? "processing" : message;
        }
    }

    private void Update()
    {
        if (box == null || !box.IsReady)
        {
            return;
        }

        UpdatePanelPlacement();
    }

    private void OnDestroy()
    {
        DestroyMaterial(fillMaterial);
        DestroyMaterial(lineMaterial);
        DestroyMaterial(panelMaterial);
    }

    private void Rebuild()
    {
        for (int i = transform.childCount - 1; i >= 0; i--)
        {
            Destroy(transform.GetChild(i).gameObject);
        }

        if (box == null || !box.IsReady)
        {
            return;
        }

        ComputeCorners();
        EnsureMaterials();
        CreateFillBox();
        CreateWireframe();
        CreatePanel();
        UpdatePanelPlacement(true);
    }

    private void ComputeCorners()
    {
        Vector3 min = box.AabbMinWorld;
        Vector3 max = box.AabbMaxWorld;
        corners[0] = new Vector3(min.x, min.y, min.z);
        corners[1] = new Vector3(max.x, min.y, min.z);
        corners[2] = new Vector3(max.x, max.y, min.z);
        corners[3] = new Vector3(min.x, max.y, min.z);
        corners[4] = new Vector3(min.x, min.y, max.z);
        corners[5] = new Vector3(max.x, min.y, max.z);
        corners[6] = new Vector3(max.x, max.y, max.z);
        corners[7] = new Vector3(min.x, max.y, max.z);
    }

    private void EnsureMaterials()
    {
        if (fillMaterial == null)
        {
            fillMaterial = CreateTransparentMaterial(new Color(0.05f, 0.62f, 1.0f, 0.18f));
        }
        if (lineMaterial == null)
        {
            Shader shader = FindFirstAvailableShader("Sprites/Default", "Standard", "Unlit/Color");
            lineMaterial = new Material(shader);
            SetMaterialColor(lineMaterial, new Color(0.05f, 1.0f, 0.95f, 0.95f));
        }
        if (panelMaterial == null)
        {
            panelMaterial = CreateTransparentMaterial(new Color(1.0f, 1.0f, 1.0f, 0.72f));
        }
    }

    private void CreateFillBox()
    {
        fillObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
        fillObject.name = "sam3_spatial_box_fill";
        fillObject.transform.SetParent(transform, false);
        fillObject.transform.position = box.CenterWorld;
        fillObject.transform.localScale = box.SizeWorld;
        Collider collider = fillObject.GetComponent<Collider>();
        if (collider != null)
        {
            Destroy(collider);
        }
        Renderer renderer = fillObject.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.sharedMaterial = fillMaterial;
        }
    }

    private void CreateWireframe()
    {
        for (int i = 0; i < EdgePairs.GetLength(0); i++)
        {
            GameObject edgeObject = new GameObject("sam3_spatial_box_edge_" + i.ToString(CultureInfo.InvariantCulture));
            edgeObject.transform.SetParent(transform, false);
            LineRenderer line = edgeObject.AddComponent<LineRenderer>();
            line.useWorldSpace = true;
            line.positionCount = 2;
            line.material = lineMaterial;
            line.startWidth = 0.008f;
            line.endWidth = 0.008f;
            line.startColor = new Color(0.05f, 1.0f, 0.95f, 0.95f);
            line.endColor = new Color(0.05f, 1.0f, 0.95f, 0.95f);
            line.numCapVertices = 2;
            line.SetPosition(0, corners[EdgePairs[i, 0]]);
            line.SetPosition(1, corners[EdgePairs[i, 1]]);
        }
    }

    private void CreatePanel()
    {
        GameObject panelObject = new GameObject("sam3_spatial_progress_panel");
        panelObject.transform.SetParent(transform, false);
        panelRoot = panelObject.transform;

        GameObject background = GameObject.CreatePrimitive(PrimitiveType.Quad);
        background.name = "background";
        background.transform.SetParent(panelRoot, false);
        background.transform.localScale = new Vector3(0.36f, 0.12f, 1.0f);
        Collider collider = background.GetComponent<Collider>();
        if (collider != null)
        {
            Destroy(collider);
        }
        Renderer renderer = background.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.sharedMaterial = panelMaterial;
        }

        GameObject textObject = new GameObject("text");
        textObject.transform.SetParent(panelRoot, false);
        textObject.transform.localPosition = new Vector3(0f, 0f, 0.006f);
        panelText = textObject.AddComponent<TextMesh>();
        panelText.text = "processing";
        panelText.anchor = TextAnchor.MiddleCenter;
        panelText.alignment = TextAlignment.Center;
        panelText.characterSize = 0.025f;
        panelText.fontSize = 48;
        panelText.color = new Color(0.02f, 0.03f, 0.04f, 1.0f);
    }

    private void UpdatePanelPlacement(bool force = false)
    {
        if (panelRoot == null)
        {
            return;
        }

        Vector3 normal = ResolveViewerFaceNormal();
        if (!faceInitialized)
        {
            activeFaceNormal = normal;
            pendingFaceNormal = normal;
            pendingFaceSince = Time.time;
            faceInitialized = true;
        }
        else if (Vector3.Dot(normal, activeFaceNormal) < 0.82f)
        {
            if (Vector3.Dot(normal, pendingFaceNormal) < 0.98f)
            {
                pendingFaceNormal = normal;
                pendingFaceSince = Time.time;
            }
            else if (Time.time - pendingFaceSince >= FaceSwitchHoldSeconds)
            {
                activeFaceNormal = normal;
            }
        }

        Vector3 targetPosition = FaceCenter(activeFaceNormal) + activeFaceNormal * PanelOffsetMeters;
        panelRoot.position = force
            ? targetPosition
            : Vector3.Lerp(panelRoot.position, targetPosition, Time.deltaTime * PanelLerpSpeed);

        Camera cam = Camera.main;
        if (cam != null)
        {
            Vector3 toCamera = panelRoot.position - cam.transform.position;
            if (toCamera.sqrMagnitude > 0.0001f)
            {
                panelRoot.rotation = Quaternion.LookRotation(toCamera.normalized, Vector3.up);
            }
        }
    }

    private Vector3 ResolveViewerFaceNormal()
    {
        Camera cam = Camera.main;
        Vector3 direction = cam != null
            ? cam.transform.position - box.CenterWorld
            : Vector3.forward;
        if (direction.sqrMagnitude < 0.0001f)
        {
            direction = Vector3.forward;
        }
        direction.Normalize();

        Vector3 abs = new Vector3(Mathf.Abs(direction.x), Mathf.Abs(direction.y), Mathf.Abs(direction.z));
        if (abs.x >= abs.y && abs.x >= abs.z)
        {
            return new Vector3(Mathf.Sign(direction.x), 0f, 0f);
        }
        if (abs.y >= abs.z)
        {
            return new Vector3(0f, Mathf.Sign(direction.y), 0f);
        }
        return new Vector3(0f, 0f, Mathf.Sign(direction.z));
    }

    private Vector3 FaceCenter(Vector3 normal)
    {
        Vector3 extents = box.SizeWorld * 0.5f;
        Vector3 offset = new Vector3(normal.x * extents.x, normal.y * extents.y, normal.z * extents.z);
        return box.CenterWorld + offset;
    }

    private static Material CreateTransparentMaterial(Color color)
    {
        Shader shader = FindFirstAvailableShader("Standard", "Unlit/Color", "Sprites/Default");
        Material material = new Material(shader);
        SetMaterialColor(material, color);
        MakeTransparent(material);
        return material;
    }

    private static Shader FindFirstAvailableShader(params string[] names)
    {
        foreach (string name in names)
        {
            Shader shader = Shader.Find(name);
            if (shader != null)
            {
                return shader;
            }
        }
        return Shader.Find("Hidden/InternalErrorShader");
    }

    private static void SetMaterialColor(Material material, Color color)
    {
        if (material == null)
        {
            return;
        }

        if (material.HasProperty("_Color"))
        {
            material.SetColor("_Color", color);
        }
        if (material.HasProperty("_BaseColor"))
        {
            material.SetColor("_BaseColor", color);
        }
    }

    private static void MakeTransparent(Material material)
    {
        if (material == null)
        {
            return;
        }

        material.SetOverrideTag("RenderType", "Transparent");
        material.SetFloat("_Mode", 3f);
        material.SetFloat("_Surface", 1f);
        material.SetFloat("_AlphaClip", 0f);
        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        material.SetInt("_ZWrite", 0);
        material.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
        material.DisableKeyword("_ALPHATEST_ON");
        material.EnableKeyword("_ALPHABLEND_ON");
        material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        material.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
    }

    private static void DestroyMaterial(Material material)
    {
        if (material != null)
        {
            Destroy(material);
        }
    }
}
