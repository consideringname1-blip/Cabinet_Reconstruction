using System.Globalization;
using UnityEngine;

[DisallowMultipleComponent]
public class RuntimeSpatialBoxDisplay : MonoBehaviour
{
    private const float EdgeDiameterMeters = 0.005f;

    // Canonical corner order:
    // [000, 100, 110, 010, 001, 101, 111, 011].
    private static readonly int[,] EdgePairs =
    {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
        { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
        { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
    };

    private RuntimeSpatialBoxData box;
    private Material edgeMaterial;
    private readonly GameObject[] edgeObjects = new GameObject[12];

    public void Configure(RuntimeSpatialBoxData spatialBox, string message)
    {
        box = spatialBox;
        Rebuild();
    }

    public void UpdateMessage(string message)
    {
        // Spatial boxes are deliberately geometry-only. Progress text belongs
        // to the regular UI and must not add filled/panel geometry to this box.
    }

    private void OnDestroy()
    {
        if (edgeMaterial != null)
        {
            Destroy(edgeMaterial);
            edgeMaterial = null;
        }
    }

    private void Rebuild()
    {
        if (box == null || !box.IsReady || !box.HasFiniteCorners)
        {
            SetAllEdgesActive(false);
            return;
        }

        EnsureMaterial();
        for (int i = 0; i < EdgePairs.GetLength(0); i++)
        {
            CreateOrUpdateEdge(
                i,
                box.CornersWorld[EdgePairs[i, 0]],
                box.CornersWorld[EdgePairs[i, 1]]);
        }
    }

    private void CreateOrUpdateEdge(int edgeIndex, Vector3 start, Vector3 end)
    {
        Vector3 delta = end - start;
        float length = delta.magnitude;
        if (!IsFinite(start) || !IsFinite(end) || length <= 0.000001f)
        {
            if (edgeObjects[edgeIndex] != null)
            {
                edgeObjects[edgeIndex].SetActive(false);
            }
            return;
        }

        GameObject edge = edgeObjects[edgeIndex];
        if (edge == null)
        {
            edge = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            edge.name = "spatial_box_edge_"
                + edgeIndex.ToString(CultureInfo.InvariantCulture);
            edge.transform.SetParent(transform, true);
            Collider collider = edge.GetComponent<Collider>();
            if (collider != null)
            {
                Destroy(collider);
            }
            Renderer renderer = edge.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = edgeMaterial;
            }
            edgeObjects[edgeIndex] = edge;
        }
        edge.SetActive(true);
        edge.transform.position = (start + end) * 0.5f;
        edge.transform.rotation = Quaternion.FromToRotation(Vector3.up, delta / length);
        // Unity's primitive cylinder has radius 0.5 and height 2.
        edge.transform.localScale = new Vector3(
            EdgeDiameterMeters,
            length * 0.5f,
            EdgeDiameterMeters);
    }

    private void SetAllEdgesActive(bool active)
    {
        foreach (GameObject edge in edgeObjects)
        {
            if (edge != null)
            {
                edge.SetActive(active);
            }
        }
    }

    private void EnsureMaterial()
    {
        if (edgeMaterial != null)
        {
            return;
        }
        Shader shader = Shader.Find("Unlit/Color");
        if (shader == null)
        {
            shader = Shader.Find("Standard");
        }
        if (shader == null)
        {
            shader = Shader.Find("Hidden/InternalErrorShader");
        }
        edgeMaterial = new Material(shader);
        Color color = new Color(0.05f, 1.0f, 0.95f, 0.95f);
        if (edgeMaterial.HasProperty("_Color"))
        {
            edgeMaterial.SetColor("_Color", color);
        }
        if (edgeMaterial.HasProperty("_BaseColor"))
        {
            edgeMaterial.SetColor("_BaseColor", color);
        }
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
}
