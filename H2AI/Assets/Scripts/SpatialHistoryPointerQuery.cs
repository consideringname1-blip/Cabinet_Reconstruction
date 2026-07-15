using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using System.Collections;
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
    [SerializeField] private bool showRayLine = true;
    [SerializeField] private Material lineMaterial;
    [SerializeField, Min(0.001f)] private float lineWidth = 0.008f;
    [SerializeField] private Color rayValidColor = new Color(0.1f, 0.9f, 1f, 1f);
    [SerializeField] private Color rayInvalidColor = new Color(1f, 0.15f, 0.1f, 1f);

    private Coroutine queryCoroutine;
    private LineRenderer rayLineRenderer;
    private Material runtimeLineMaterial;

    private void Awake()
    {
        EnsureLineMaterial();
        if (showRayLine)
        {
            EnsureRayLine();
        }
    }

    private void Update()
    {
        if (!showRayLine)
        {
            if (rayLineRenderer != null)
            {
                rayLineRenderer.gameObject.SetActive(false);
            }
            return;
        }

        EnsureRayLine();
        if (!TryGetFingerRay(out Vector3 origin, out Vector3 direction))
        {
            rayLineRenderer.gameObject.SetActive(false);
            return;
        }

        bool hitModel = TryFindPointedModel(origin, direction, out _, out Vector3 endpoint);
        rayLineRenderer.gameObject.SetActive(true);
        rayLineRenderer.SetPosition(0, origin);
        rayLineRenderer.SetPosition(1, endpoint);
        Color color = hitModel ? rayValidColor : rayInvalidColor;
        rayLineRenderer.startColor = color;
        rayLineRenderer.endColor = color;
    }

    public void TriggerSpatialHistoryQuery()
    {
        if (queryCoroutine != null)
        {
            StopCoroutine(queryCoroutine);
        }
        queryCoroutine = StartCoroutine(QueryPointedObject());
    }

    private IEnumerator QueryPointedObject()
    {
        if (waitSeconds > 0f)
        {
            yield return new WaitForSeconds(waitSeconds);
        }
        queryCoroutine = null;

        if (!TryGetFingerRay(out Vector3 origin, out Vector3 direction))
        {
            ShowFrontMessage("spatial_query_ERR_hand_not_tracked");
            yield break;
        }
        if (!TryFindPointedModel(origin, direction, out RuntimeModelEventIdentity identity, out _)
            || identity == null
            || string.IsNullOrEmpty(identity.DisplayObjectId))
        {
            ShowFrontMessage("spatial_query_no_hit");
            yield break;
        }

        HistoryPresentationController controller = HistoryPresentationController.Instance;
        if (controller == null)
        {
            ShowFrontMessage("history_presentation_controller_missing");
            yield break;
        }
        controller.OnModelClicked(identity.DisplayObjectId);
    }

    private bool TryGetFingerRay(out Vector3 origin, out Vector3 direction)
    {
        origin = Vector3.zero;
        direction = Vector3.forward;
        Handedness handedness = targetHand == QueryHand.Right
            ? Handedness.Right
            : Handedness.Left;
        if (!HandJointUtils.TryGetJointPose(
                TrackedHandJoint.IndexTip,
                handedness,
                out MixedRealityPose tip)
            || !HandJointUtils.TryGetJointPose(
                TrackedHandJoint.IndexDistalJoint,
                handedness,
                out MixedRealityPose distal))
        {
            return false;
        }

        Vector3 rayDirection = tip.Position - distal.Position;
        if (rayDirection.sqrMagnitude < 0.000001f)
        {
            rayDirection = tip.Rotation * Vector3.forward;
        }
        if (rayDirection.sqrMagnitude < 0.000001f)
        {
            return false;
        }
        origin = tip.Position;
        direction = rayDirection.normalized;
        return true;
    }

    private bool TryFindPointedModel(
        Vector3 origin,
        Vector3 direction,
        out RuntimeModelEventIdentity identity,
        out Vector3 endpoint)
    {
        identity = null;
        float distance = Mathf.Max(0.1f, maxDistanceMeters);
        endpoint = origin + direction * distance;
        Ray ray = new Ray(origin, direction);
        float nearestDistance = distance;
        RuntimeModelEventIdentity[] candidates =
            FindObjectsOfType<RuntimeModelEventIdentity>();
        foreach (RuntimeModelEventIdentity candidate in candidates)
        {
            if (candidate == null
                || !candidate.gameObject.activeInHierarchy
                || string.IsNullOrEmpty(candidate.DisplayObjectId)
                || !candidate.TryGetWorldBounds(out Bounds bounds)
                || !bounds.IntersectRay(ray, out float hitDistance)
                || hitDistance < 0f
                || hitDistance > nearestDistance)
            {
                continue;
            }
            identity = candidate;
            nearestDistance = hitDistance;
        }
        if (identity == null)
        {
            return false;
        }
        endpoint = ray.GetPoint(nearestDistance);
        return true;
    }

    private void EnsureLineMaterial()
    {
        if (runtimeLineMaterial != null)
        {
            return;
        }
        if (lineMaterial != null)
        {
            runtimeLineMaterial = lineMaterial;
            return;
        }
        Shader shader = Shader.Find("Sprites/Default");
        runtimeLineMaterial = new Material(
            shader != null ? shader : Shader.Find("Standard"));
    }

    private void EnsureRayLine()
    {
        if (rayLineRenderer != null)
        {
            return;
        }
        EnsureLineMaterial();
        GameObject rayObject = new GameObject("SpatialHistoryPointerRay");
        rayObject.transform.SetParent(transform, false);
        rayLineRenderer = rayObject.AddComponent<LineRenderer>();
        rayLineRenderer.useWorldSpace = true;
        rayLineRenderer.positionCount = 2;
        rayLineRenderer.material = runtimeLineMaterial;
        rayLineRenderer.startWidth = lineWidth;
        rayLineRenderer.endWidth = lineWidth;
        rayLineRenderer.numCapVertices = 4;
    }

    private void ShowFrontMessage(string message)
    {
        if (Game_M.initialize != null)
        {
            Game_M.initialize.XianShi(message);
        }
    }
}
