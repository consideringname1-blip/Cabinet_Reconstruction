using UnityEngine;

public class SelectionBoxDebugActions : MonoBehaviour
{
    [SerializeField] private GameObject selectionBoxRoot;
    [SerializeField] private float distanceFromCamera = 0.8f;
    [SerializeField] private bool hideOnStart = true;

    [SerializeField] private GameObject settingsPanel;
    [SerializeField] private float settingsDistanceFromCamera = 1.0f;
    [SerializeField] private bool hideSettingsOnStart = true;

    private void Awake()
    {
        if (hideOnStart && selectionBoxRoot != null)
        {
            selectionBoxRoot.SetActive(false);
        }

        if (hideSettingsOnStart && settingsPanel != null)
        {
            settingsPanel.SetActive(false);
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

        PlaceSelectionBoxInFrontOfCamera();
        selectionBoxRoot.SetActive(true);
    }

    public void ToggleSettingsPanel()
    {
        if (settingsPanel == null)
        {
            Debug.LogWarning("[SelectionBoxDebugActions] settingsPanel is not assigned.");
            return;
        }

        if (settingsPanel.activeSelf)
        {
            settingsPanel.SetActive(false);
            return;
        }

        Camera mainCamera = Camera.main;
        if (mainCamera == null)
        {
            Debug.LogWarning("[SelectionBoxDebugActions] Main Camera not found.");
            return;
        }

        Transform cameraTransform = mainCamera.transform;
        Vector3 panelPosition = cameraTransform.position + cameraTransform.forward * Mathf.Max(0.1f, settingsDistanceFromCamera);
        Quaternion panelRotation = Quaternion.LookRotation(cameraTransform.forward, Vector3.up);

        settingsPanel.transform.SetPositionAndRotation(panelPosition, panelRotation);
        settingsPanel.SetActive(true);
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
}
