using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Displays the current headset (usually Camera.main) world pose in real time.
/// Attach this to any GameObject and optionally assign a TMP_Text or UI Text.
/// </summary>
[DisallowMultipleComponent]
public class HeadsetPoseDisplay : MonoBehaviour
{
    [Header("Source")]
    [SerializeField] private Transform targetTransform;
    [SerializeField] private bool useMainCameraIfMissing = true;
    [SerializeField] private bool driveThisGameObjectInFrontOfHeadset = true;
    [SerializeField] private float forwardDistance = 1.0f;
    [SerializeField] private bool faceBackTowardHeadset = true;

    [Header("Output")]
    [SerializeField] private TMP_Text tmpText;
    [SerializeField] private Text uiText;
    [SerializeField] private bool writeToGameObjectName = true;
    [SerializeField] private bool includeRotation = true;
    [SerializeField] private int decimals = 3;

    [Header("Debug")]
    [TextArea(3, 8)]
    [SerializeField] private string latestDisplayText;

    private void Awake()
    {
        ResolveTargetIfNeeded();
        UpdateDisplay();
    }

    private void Update()
    {
        ResolveTargetIfNeeded();
        UpdateDisplay();
    }

    private void ResolveTargetIfNeeded()
    {
        if (targetTransform != null || !useMainCameraIfMissing)
        {
            return;
        }

        Camera mainCam = Camera.main;
        if (mainCam != null)
        {
            targetTransform = mainCam.transform;
        }
    }

    private void UpdateDisplay()
    {
        if (targetTransform == null)
        {
            latestDisplayText = "Headset: target not found";
            ApplyText(latestDisplayText);
            return;
        }

        Vector3 pos = targetTransform.position;
        Vector3 rot = targetTransform.eulerAngles;

        if (driveThisGameObjectInFrontOfHeadset)
        {
            Vector3 targetPosition = targetTransform.position + targetTransform.forward * forwardDistance;
            Quaternion targetRotation = faceBackTowardHeadset
                ? Quaternion.LookRotation(targetTransform.position - targetPosition, Vector3.up)
                : targetTransform.rotation;
            transform.SetPositionAndRotation(targetPosition, targetRotation);
        }

        string format = "F" + Mathf.Clamp(decimals, 0, 6);
        latestDisplayText =
            "Headset World Pose\n" +
            $"Position: ({pos.x.ToString(format)}, {pos.y.ToString(format)}, {pos.z.ToString(format)})";

        if (includeRotation)
        {
            latestDisplayText +=
                $"\nRotation: ({rot.x.ToString(format)}, {rot.y.ToString(format)}, {rot.z.ToString(format)})";
        }

        ApplyText(latestDisplayText);
    }

    private void ApplyText(string value)
    {
        if (writeToGameObjectName)
        {
            gameObject.name = value.Replace('\n', ' ');
        }

        if (tmpText != null)
        {
            tmpText.text = value;
        }

        if (uiText != null)
        {
            uiText.text = value;
        }
    }
}
