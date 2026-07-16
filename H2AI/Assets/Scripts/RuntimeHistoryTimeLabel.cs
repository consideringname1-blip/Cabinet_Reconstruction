using UnityEngine;

[DisallowMultipleComponent]
public class RuntimeHistoryTimeLabel : MonoBehaviour
{
    private const float MinimumVerticalOffsetMeters = 0.06f;
    private const float BoundsOffsetRatio = 0.15f;

    private GameObject modelRoot;
    private TextMesh textMesh;

    public void Configure(GameObject targetModelRoot, string displayTimeJst)
    {
        modelRoot = targetModelRoot;
        EnsureTextMesh();
        textMesh.text = displayTimeJst ?? "";
        UpdatePlacement();
    }

    private void Awake()
    {
        EnsureTextMesh();
    }

    private void LateUpdate()
    {
        UpdatePlacement();
    }

    private void EnsureTextMesh()
    {
        if (textMesh == null)
        {
            textMesh = GetComponent<TextMesh>();
        }
        if (textMesh == null)
        {
            textMesh = gameObject.AddComponent<TextMesh>();
        }
        textMesh.anchor = TextAnchor.MiddleCenter;
        textMesh.alignment = TextAlignment.Center;
        textMesh.fontSize = 64;
        textMesh.characterSize = 0.0025f;
        textMesh.richText = false;
        textMesh.color = new Color(0.05f, 1.0f, 0.95f, 1.0f);
    }

    private void UpdatePlacement()
    {
        if (modelRoot == null || textMesh == null)
        {
            return;
        }

        Vector3 anchor = modelRoot.transform.position +
            Vector3.up * MinimumVerticalOffsetMeters;
        Renderer[] renderers =
            modelRoot.GetComponentsInChildren<Renderer>(true);
        if (renderers.Length > 0)
        {
            Bounds bounds = renderers[0].bounds;
            for (int i = 1; i < renderers.Length; i++)
            {
                bounds.Encapsulate(renderers[i].bounds);
            }
            float offset = Mathf.Max(
                MinimumVerticalOffsetMeters,
                bounds.extents.y * BoundsOffsetRatio);
            anchor = new Vector3(
                bounds.center.x,
                bounds.max.y + offset,
                bounds.center.z);
        }
        transform.position = anchor;

        Camera camera = Camera.main;
        if (camera == null)
        {
            return;
        }
        Vector3 awayFromCamera =
            transform.position - camera.transform.position;
        if (awayFromCamera.sqrMagnitude > 0.0001f)
        {
            transform.rotation = Quaternion.LookRotation(
                awayFromCamera.normalized,
                Vector3.up);
        }
    }
}
