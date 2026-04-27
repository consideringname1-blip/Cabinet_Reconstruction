using UnityEngine;

public class SettingsPanelToggle : MonoBehaviour
{
    [SerializeField] private GameObject settingsPanel;
    [SerializeField] private float distanceFromCamera = 1.0f;
    [SerializeField] private bool hideOnStart = true;

    private void Awake()
    {
        if (hideOnStart && settingsPanel != null)
        {
            settingsPanel.SetActive(false);
        }
    }

    public void ToggleSettingsPanel()
    {
        if (settingsPanel == null)
        {
            Debug.LogWarning("[SettingsPanelToggle] settingsPanel is not assigned.");
            return;
        }

        bool shouldShow = !settingsPanel.activeSelf;
        if (!shouldShow)
        {
            settingsPanel.SetActive(false);
            return;
        }

        Camera mainCamera = Camera.main;
        if (mainCamera == null)
        {
            Debug.LogWarning("[SettingsPanelToggle] Main Camera not found.");
            return;
        }

        Transform cameraTransform = mainCamera.transform;
        Vector3 panelPosition = cameraTransform.position + cameraTransform.forward * distanceFromCamera;
        Quaternion panelRotation = Quaternion.LookRotation(cameraTransform.forward, Vector3.up);

        settingsPanel.transform.SetPositionAndRotation(panelPosition, panelRotation);
        settingsPanel.SetActive(true);
    }
}
