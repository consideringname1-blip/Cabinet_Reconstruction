using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class SelectionPanelManager : MonoBehaviour
{
    [Header("Panel References")]
    [SerializeField] private GameObject panelRoot;
    [SerializeField] private RawImage previewRawImage;
    [SerializeField] private SelectionBoxController selectionBoxController;
    [SerializeField] private SelectionButtonsUI selectionButtonsUI;

    [Header("Options")]
    [SerializeField] private bool hidePanelOnStart = true;
    [SerializeField] private float panelDistance = 1.0f;
    [SerializeField] private float panelVerticalOffset = 0.15f;
    [SerializeField] private float panelHorizontalOffset = 0.0f;

    public bool IsBusy => isBusy;
    public bool LastConfirmed { get; private set; }
    public bool LastForceRebuild { get; private set; }
    public Vector2 LastTopLeftNormalized { get; private set; } = Vector2.zero;
    public Vector2 LastBottomRightNormalized { get; private set; } = Vector2.zero;

    private bool isBusy;
    private bool waitFinished;

    private void Awake()
    {
        if (hidePanelOnStart && panelRoot != null)
        {
            panelRoot.SetActive(false);
        }
    }

    private void OnEnable()
    {
        SubscribeButtonEvents();
    }

    private void OnDisable()
    {
        UnsubscribeButtonEvents();
    }

    private void SubscribeButtonEvents()
    {
        if (selectionButtonsUI == null)
        {
            Debug.LogWarning("[SelectionPanelManager] selectionButtonsUI is not assigned.", this);
            return;
        }

        selectionButtonsUI.ConfirmClicked -= HandleConfirmClicked;
        selectionButtonsUI.CancelClicked -= HandleCancelClicked;
        selectionButtonsUI.ConfirmClicked += HandleConfirmClicked;
        selectionButtonsUI.CancelClicked += HandleCancelClicked;
    }

    private void UnsubscribeButtonEvents()
    {
        if (selectionButtonsUI == null)
        {
            return;
        }

        selectionButtonsUI.ConfirmClicked -= HandleConfirmClicked;
        selectionButtonsUI.CancelClicked -= HandleCancelClicked;
    }

    private bool IsReady()
    {
        if (panelRoot == null)
        {
            ShowFrontMessage("spm_ERR_ready_panelRoot_null");
            return false;
        }

        if (previewRawImage == null)
        {
            ShowFrontMessage("spm_ERR_ready_previewRawImage_null");
            return false;
        }

        if (selectionBoxController == null)
        {
            ShowFrontMessage("spm_ERR_ready_selectionBoxController_null");
            return false;
        }

        if (selectionButtonsUI == null)
        {
            ShowFrontMessage("spm_ERR_ready_selectionButtonsUI_null");
            return false;
        }

        return true;
    }

    private void ResetResultState()
    {
        LastConfirmed = false;
        LastForceRebuild = false;
        LastTopLeftNormalized = Vector2.zero;
        LastBottomRightNormalized = Vector2.zero;
    }

    private void HandleConfirmClicked(bool forceRebuild)
    {
        if (!isBusy)
        {
            return;
        }

        CaptureCurrentResult(true, forceRebuild);
        waitFinished = true;
    }

    private void HandleCancelClicked()
    {
        if (!isBusy)
        {
            return;
        }

        CaptureCurrentResult(false, false);
        waitFinished = true;
    }

    private void CaptureCurrentResult(bool confirmed, bool forceRebuild)
    {
        LastConfirmed = confirmed;
        LastForceRebuild = confirmed && forceRebuild;
        selectionBoxController.GetNormalizedTLBR(
            out Vector2 topLeft,
            out Vector2 bottomRight
        );
        LastTopLeftNormalized = topLeft;
        LastBottomRightNormalized = bottomRight;
    }

    public IEnumerator RequestSelection(Texture2D texture, Transform cameraTransform)
    {
        if (cameraTransform == null)
        {
            ShowFrontMessage("spm_ERR_req1_cameraTransform_null");
            yield break;
        }

        if (texture == null)
        {
            ShowFrontMessage("spm_ERR_req1_texture_null");
            yield break;
        }

        yield return StartCoroutine(
            RequestSelection(texture, cameraTransform.position, cameraTransform.rotation)
        );
    }

    public IEnumerator RequestSelection(Texture2D texture, Vector3 cameraPosition, Quaternion cameraRotation)
    {
        if (!IsReady())
        {
            ShowFrontMessage("spm_ERR_req2_not_ready");
            yield break;
        }

        if (isBusy)
        {
            ShowFrontMessage("spm_ERR_req2_busy");
            yield break;
        }

        isBusy = true;
        waitFinished = false;
        ResetResultState();

        PlacePanelInFrontOfCamera(cameraPosition, cameraRotation, panelDistance);
        panelRoot.SetActive(true);

        previewRawImage.texture = texture;
        previewRawImage.color = Color.white;

        yield return null;

        selectionBoxController.PrepareForReuse();
        yield return new WaitUntil(() => waitFinished);

        panelRoot.SetActive(false);
        isBusy = false;
    }

    private void PlacePanelInFrontOfCamera(Vector3 cameraPosition, Quaternion cameraRotation, float distance)
    {
        if (panelRoot == null)
        {
            return;
        }

        Vector3 flatForward = Vector3.ProjectOnPlane(cameraRotation * Vector3.forward, Vector3.up);
        if (flatForward.sqrMagnitude < 1e-6f)
        {
            flatForward = Vector3.forward;
        }
        flatForward.Normalize();

        Vector3 flatRight = Vector3.Cross(Vector3.up, flatForward).normalized;
        Vector3 panelPosition =
            cameraPosition
            + flatForward * distance
            + Vector3.up * panelVerticalOffset
            + flatRight * panelHorizontalOffset;
        Quaternion panelRotation = Quaternion.LookRotation(flatForward, Vector3.up);

        panelRoot.transform.SetPositionAndRotation(panelPosition, panelRotation);
    }

    private void ShowFrontMessage(string message)
    {
        if (Game_M.initialize != null)
        {
            Game_M.initialize.XianShi(message);
        }
    }
}
