using UnityEngine;

public class SelectionBoxDebugActions : MonoBehaviour
{
    [SerializeField] private GameObject selectionBoxRoot;
    [SerializeField] private SelectionBoxController selectionBoxController;
    [SerializeField] private float distanceFromCamera = 0.8f;
    [SerializeField] private bool hideOnStart = true;

    private void Awake()
    {
        if (hideOnStart && selectionBoxRoot != null)
        {
            selectionBoxRoot.SetActive(false);
        }
    }

    public void MoveSelectionBoxInFrontOfCamera()
    {
        if (selectionBoxRoot == null)
        {
            Debug.LogWarning("[SelectionBoxDebugActions] selectionBoxRoot is not assigned.");
            return;
        }

        PlaceSelectionBoxInFrontOfCamera();
        selectionBoxRoot.SetActive(true);
    }

    public void ResetSelectionBoxInPlace()
    {
        SelectionBoxController controller = ResolveSelectionBoxController();
        if (controller == null)
        {
            Debug.LogWarning("[SelectionBoxDebugActions] SelectionBoxController was not found.");
            return;
        }

        if (selectionBoxRoot != null)
        {
            PlaceSelectionBoxInFrontOfCamera();
            selectionBoxRoot.SetActive(true);
        }

        controller.PrepareForReuse();
    }

    private bool PlaceSelectionBoxInFrontOfCamera()
    {
        if (selectionBoxRoot == null)
        {
            return false;
        }

        Camera mainCamera = Camera.main;
        if (mainCamera == null)
        {
            Debug.LogWarning("[SelectionBoxDebugActions] Main Camera not found.");
            return false;
        }

        Transform cameraTransform = mainCamera.transform;
        Vector3 flatForward = Vector3.ProjectOnPlane(cameraTransform.forward, Vector3.up);
        if (flatForward.sqrMagnitude < 0.000001f)
        {
            flatForward = Vector3.forward;
        }
        flatForward.Normalize();

        Vector3 targetPosition = cameraTransform.position + flatForward * Mathf.Max(0.1f, distanceFromCamera);
        Quaternion targetRotation = Quaternion.LookRotation(flatForward, Vector3.up);
        selectionBoxRoot.transform.SetPositionAndRotation(targetPosition, targetRotation);
        return true;
    }

    private SelectionBoxController ResolveSelectionBoxController()
    {
        if (selectionBoxController != null)
        {
            return selectionBoxController;
        }

        if (selectionBoxRoot != null)
        {
            selectionBoxController = selectionBoxRoot.GetComponentInChildren<SelectionBoxController>(true);
        }

        return selectionBoxController;
    }
}
