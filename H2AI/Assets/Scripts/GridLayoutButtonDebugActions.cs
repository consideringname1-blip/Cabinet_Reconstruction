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

        if (selectionBoxRoot.activeSelf)
        {
            selectionBoxRoot.SetActive(false);
            return;
        }

        Camera mainCamera = Camera.main;
        if (mainCamera == null)
        {
            Debug.LogWarning("[SelectionBoxDebugActions] Main Camera not found.");
            return;
        }

        Transform cameraTransform = mainCamera.transform;
        Vector3 targetPosition = cameraTransform.position + cameraTransform.forward * distanceFromCamera;
        Quaternion targetRotation = Quaternion.LookRotation(cameraTransform.forward, Vector3.up);

        selectionBoxRoot.transform.SetPositionAndRotation(targetPosition, targetRotation);
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

        if (selectionBoxRoot != null && !selectionBoxRoot.activeSelf)
        {
            selectionBoxRoot.SetActive(true);
        }

        controller.PrepareForReuse();
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
