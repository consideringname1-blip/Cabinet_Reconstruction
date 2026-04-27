using UnityEngine;

public class SelectionBoxDebugActions : MonoBehaviour
{
    [SerializeField] private GameObject selectionBoxRoot;
    [SerializeField] private float distanceFromCamera = 0.8f;

    public void MoveSelectionBoxInFrontOfCamera()
    {
        if (selectionBoxRoot == null)
        {
            Debug.LogWarning("[SelectionBoxDebugActions] selectionBoxRoot is not assigned.");
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
    }
}
