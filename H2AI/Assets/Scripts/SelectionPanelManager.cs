using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using UnityEngine.UI;


public class SelectionPanelManager : MonoBehaviour
{
    [Header("Panel References")]
    [SerializeField] private GameObject panelRoot;                  // Canvas Selection box
    [SerializeField] private RawImage previewRawImage;              // RawImage Selection box
    [SerializeField] private SelectionBoxController selectionBoxController;
    [SerializeField] private SelectionButtonsUI selectionButtonsUI;

    [Header("Options")]
    [SerializeField] private bool hidePanelOnStart = true;
    [SerializeField] private float panelDistance = 1.0f;            // 面板放在相机水平前方 1m
    [SerializeField] private float panelVerticalOffset = 0.15f;   // 往上抬 15 cm
    [SerializeField] private float panelHorizontalOffset = 0.0f;  // 向右为正，向左为负

    public bool IsBusy => isBusy;

    // 协程结束后，由外部读取这几个结果
    public bool LastConfirmed { get; private set; } = false;

    // 左上角 / 右下角，都是相对图片左上角的归一化比例坐标
    // x 向右增大，y 向下增大，范围 0~1
    public Vector2 LastTopLeftNormalized { get; private set; } = Vector2.zero;
    public Vector2 LastBottomRightNormalized { get; private set; } = Vector2.zero;

    private bool isBusy = false;
    private bool waitFinished = false;

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
        if (selectionButtonsUI == null) return;

        selectionButtonsUI.ConfirmClicked -= HandleConfirmClicked;
        selectionButtonsUI.CancelClicked -= HandleCancelClicked;

        selectionButtonsUI.ConfirmClicked += HandleConfirmClicked;
        selectionButtonsUI.CancelClicked += HandleCancelClicked;
    }

    private void UnsubscribeButtonEvents()
    {
        if (selectionButtonsUI == null) return;

        selectionButtonsUI.ConfirmClicked -= HandleConfirmClicked;
        selectionButtonsUI.CancelClicked -= HandleCancelClicked;
    }

    private bool IsReady()
    {
        return panelRoot != null &&
               previewRawImage != null &&
               selectionBoxController != null &&
               selectionButtonsUI != null;
    }

    private void ResetResultState()
    {
        LastConfirmed = false;
        LastTopLeftNormalized = Vector2.zero;
        LastBottomRightNormalized = Vector2.zero;
    }

    private void HandleConfirmClicked()
    {
        if (!isBusy) return;

        CaptureCurrentResult(true);
        waitFinished = true;
    }

    private void HandleCancelClicked()
    {
        if (!isBusy) return;

        CaptureCurrentResult(false);
        waitFinished = true;
    }

    private void CaptureCurrentResult(bool confirmed)
    {
        LastConfirmed = confirmed;

        // 这里直接取“相对于图片左上角的归一化坐标”
        selectionBoxController.GetNormalizedTLBR(
            out Vector2 topLeft,
            out Vector2 bottomRight
        );

        LastTopLeftNormalized = topLeft;
        LastBottomRightNormalized = bottomRight;
    }

    /// <summary>
    /// 简化调用版：
    /// 直接传相机 Transform。
    /// 面板会被放到相机水平方向前方 panelDistance 米处。
    /// </summary>
    public IEnumerator RequestSelection(Texture2D texture, Transform cameraTransform)
    {
        if (cameraTransform == null)
        {
            Debug.LogError("SelectionPanelManager: cameraTransform is null.");
            yield break;
        }

        yield return StartCoroutine(
            RequestSelection(texture, cameraTransform.position, cameraTransform.rotation)
        );
    }

    /// <summary>
    /// 主协程：
    /// 传入图片 + 相机位姿。
    /// 调用后会：
    /// 1. 打开面板
    /// 2. 显示图片
    /// 3. 初始化框
    /// 4. 等待用户点击 Confirm / Cancel
    /// 5. 保存结果到 LastConfirmed / LastTopLeftNormalized / LastBottomRightNormalized
    /// 6. 关闭面板
    /// </summary>
    public IEnumerator RequestSelection(Texture2D texture, Vector3 cameraPosition, Quaternion cameraRotation)
    {
        if (!IsReady())
        {
            Debug.LogError("SelectionPanelManager: references are not fully assigned.");
            yield break;
        }

        if (isBusy)
        {
            Debug.LogWarning("SelectionPanelManager: another selection request is already running.");
            yield break;
        }

        isBusy = true;
        waitFinished = false;
        ResetResultState();

        // 1. 根据相机位姿摆放面板
        PlacePanelInFrontOfCamera(cameraPosition, cameraRotation, panelDistance);

        // 2. 打开面板
        panelRoot.SetActive(true);

        // 3. 设置图片
        previewRawImage.texture = texture;

        // 4. 等一帧，让面板激活、子物体和布局稳定
        yield return null;

        // 5. 每次调用都重新初始化选框和两个 handle
        selectionBoxController.PrepareForReuse();

        // 6. 等待用户点击
        yield return new WaitUntil(() => waitFinished);

        // 7. 关闭面板
        panelRoot.SetActive(false);

        isBusy = false;
    }

    /// <summary>
    /// 将 panelRoot 放到“相机水平方向前方 distance 米处”，并水平朝向相机。
    /// 忽略相机抬头/低头带来的上下分量。
    /// </summary>
    private void PlacePanelInFrontOfCamera(Vector3 cameraPosition, Quaternion cameraRotation, float distance)
    {
        if (panelRoot == null) return;

        // 相机 forward 投影到水平面
        Vector3 flatForward = Vector3.ProjectOnPlane(cameraRotation * Vector3.forward, Vector3.up);

        if (flatForward.sqrMagnitude < 1e-6f)
        {
            flatForward = Vector3.ProjectOnPlane(cameraRotation * Vector3.up, Vector3.up);
        }

        if (flatForward.sqrMagnitude < 1e-6f)
        {
            flatForward = Vector3.forward;
        }

        flatForward.Normalize();

        // 水平右方向（用于左右微调）
        Vector3 flatRight = Vector3.Cross(Vector3.up, flatForward).normalized;

        // 面板位置：正前方 + 上移 + 左右微调
        Vector3 panelPosition =
            cameraPosition
            + flatForward * distance
            + Vector3.up * panelVerticalOffset
            + flatRight * panelHorizontalOffset;

        // 面板朝向相机（允许上下看向相机，就不要把 y 清零）
        Vector3 lookDir = cameraPosition - panelPosition;

        if (lookDir.sqrMagnitude < 1e-6f)
        {
            lookDir = -flatForward;
        }

        Quaternion panelRotation = Quaternion.LookRotation(lookDir.normalized, Vector3.up);

        panelRoot.transform.SetPositionAndRotation(panelPosition, panelRotation);
    }
}