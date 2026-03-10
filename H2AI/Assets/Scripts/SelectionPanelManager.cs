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
    [SerializeField] private float panelVerticalOffset = 0.15f;     // 往上抬 15 cm
    [SerializeField] private float panelHorizontalOffset = 0.0f;    // 向右为正，向左为负

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
        SubscribeButtonEvents();
        Game_M.initialize.XianShi("spm_awake_01_enter");

        if (hidePanelOnStart && panelRoot != null)
        {
            panelRoot.SetActive(false);
            Game_M.initialize.XianShi("spm_awake_02_panel_hide");
        }
        else if (panelRoot == null)
        {
            Game_M.initialize.XianShi("spm_ERR_awake_panelRoot_null");
        }
    }

    private void OnEnable()
    {
        Game_M.initialize.XianShi("spm_onenable_01");
        SubscribeButtonEvents();
    }

    private void OnDisable()
    {
        Game_M.initialize.XianShi("spm_ondisable_01");
        UnsubscribeButtonEvents();
    }

    private void SubscribeButtonEvents()
    {
        if (selectionButtonsUI == null)
        {
            Game_M.initialize.XianShi("spm_ERR_subscribe_buttonsUI_null");
            return;
        }

        selectionButtonsUI.ConfirmClicked -= HandleConfirmClicked;
        selectionButtonsUI.CancelClicked -= HandleCancelClicked;

        selectionButtonsUI.ConfirmClicked += HandleConfirmClicked;
        selectionButtonsUI.CancelClicked += HandleCancelClicked;

        Game_M.initialize.XianShi("spm_subscribe_ok");
    }

    private void UnsubscribeButtonEvents()
    {
        if (selectionButtonsUI == null)
        {
            Game_M.initialize.XianShi("spm_ERR_unsubscribe_buttonsUI_null");
            return;
        }

        selectionButtonsUI.ConfirmClicked -= HandleConfirmClicked;
        selectionButtonsUI.CancelClicked -= HandleCancelClicked;

        Game_M.initialize.XianShi("spm_unsubscribe_ok");
    }

    private bool IsReady()
    {
        if (panelRoot == null)
        {
            Game_M.initialize.XianShi("spm_ERR_ready_panelRoot_null");
            return false;
        }

        if (previewRawImage == null)
        {
            Game_M.initialize.XianShi("spm_ERR_ready_previewRawImage_null");
            return false;
        }

        if (selectionBoxController == null)
        {
            Game_M.initialize.XianShi("spm_ERR_ready_selectionBoxController_null");
            return false;
        }

        if (selectionButtonsUI == null)
        {
            Game_M.initialize.XianShi("spm_ERR_ready_selectionButtonsUI_null");
            return false;
        }

        Game_M.initialize.XianShi("spm_ready_ok");
        return true;
    }

    private void ResetResultState()
    {
        LastConfirmed = false;
        LastTopLeftNormalized = Vector2.zero;
        LastBottomRightNormalized = Vector2.zero;

        Game_M.initialize.XianShi("spm_reset_result");
    }

    private void HandleConfirmClicked()
    {
        Game_M.initialize.XianShi("spm_btn_confirm_click");

        if (!isBusy)
        {
            Game_M.initialize.XianShi("spm_btn_confirm_notbusy");
            return;
        }

        CaptureCurrentResult(true);
        waitFinished = true;

        Game_M.initialize.XianShi("spm_btn_confirm_done");
    }

    private void HandleCancelClicked()
    {
        Game_M.initialize.XianShi("spm_btn_cancel_click");

        if (!isBusy)
        {
            Game_M.initialize.XianShi("spm_btn_cancel_notbusy");
            return;
        }

        CaptureCurrentResult(false);
        waitFinished = true;

        Game_M.initialize.XianShi("spm_btn_cancel_done");
    }

    private void CaptureCurrentResult(bool confirmed)
    {
        Game_M.initialize.XianShi("spm_capture_01_enter");

        LastConfirmed = confirmed;

        selectionBoxController.GetNormalizedTLBR(
            out Vector2 topLeft,
            out Vector2 bottomRight
        );

        LastTopLeftNormalized = topLeft;
        LastBottomRightNormalized = bottomRight;

        Game_M.initialize.XianShi(
            $"spm_capture_02_ok_{(confirmed ? "confirm" : "cancel")}_TL({topLeft.x:F2},{topLeft.y:F2})_BR({bottomRight.x:F2},{bottomRight.y:F2})"
        );
    }

    /// <summary>
    /// 简化调用版：
    /// 直接传相机 Transform。
    /// 面板会被放到相机水平方向前方 panelDistance 米处。
    /// </summary>
    public IEnumerator RequestSelection(Texture2D texture, Transform cameraTransform)
    {
        Game_M.initialize.XianShi("spm_req1_01_enter");

        if (cameraTransform == null)
        {
            Game_M.initialize.XianShi("spm_ERR_req1_cameraTransform_null");
            yield break;
        }

        if (texture == null)
        {
            Game_M.initialize.XianShi("spm_ERR_req1_texture_null");
            yield break;
        }

        Game_M.initialize.XianShi("spm_req1_02_before_inner");

        yield return StartCoroutine(
            RequestSelection(texture, cameraTransform.position, cameraTransform.rotation)
        );

        Game_M.initialize.XianShi("spm_req1_03_exit");
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
        Game_M.initialize.XianShi("spm_req2_01_enter");

        if (!IsReady())
        {
            Game_M.initialize.XianShi("spm_ERR_req2_not_ready");
            yield break;
        }

        if (isBusy)
        {
            Game_M.initialize.XianShi("spm_ERR_req2_busy");
            yield break;
        }

        isBusy = true;
        waitFinished = false;
        ResetResultState();

        Game_M.initialize.XianShi("spm_req2_02_before_place");

        // 1. 根据相机位姿摆放面板
        PlacePanelInFrontOfCamera(cameraPosition, cameraRotation, panelDistance);

        Game_M.initialize.XianShi("spm_req2_03_after_place");

        // 2. 打开面板
        panelRoot.SetActive(true);

        Game_M.initialize.XianShi("spm_req2_04_after_active");

        // 3. 设置图片
        previewRawImage.texture = texture;
        previewRawImage.color = Color.white;

        Game_M.initialize.XianShi("spm_req2_05_after_texture");

        // 4. 等一帧，让面板激活、子物体和布局稳定
        yield return null;

        Game_M.initialize.XianShi("spm_req2_06_after_yield");

        // 5. 每次调用都重新初始化选框和两个 handle
        selectionBoxController.PrepareForReuse();

        Game_M.initialize.XianShi("spm_req2_07_after_prepare");

        // 6. 等待用户点击
        Game_M.initialize.XianShi("spm_req2_08_wait_click");
        yield return new WaitUntil(() => waitFinished);

        Game_M.initialize.XianShi("spm_req2_09_after_wait");

        // 7. 关闭面板
        panelRoot.SetActive(false);

        Game_M.initialize.XianShi("spm_req2_10_after_close");

        isBusy = false;

        Game_M.initialize.XianShi("spm_req2_11_finish");
    }

    /// <summary>
    /// 将 panelRoot 放到“相机水平方向前方 distance 米处”，并水平朝向前方。
    /// </summary>
    private void PlacePanelInFrontOfCamera(Vector3 cameraPosition, Quaternion cameraRotation, float distance)
    {
        Game_M.initialize.XianShi("spm_place_01_enter");

        if (panelRoot == null)
        {
            Game_M.initialize.XianShi("spm_ERR_place_panelRoot_null");
            return;
        }

        // 相机 forward 投影到水平面
        Vector3 flatForward = Vector3.ProjectOnPlane(cameraRotation * Vector3.forward, Vector3.up);

        if (flatForward.sqrMagnitude < 1e-6f)
        {
            Game_M.initialize.XianShi("spm_place_02_flatForward_fallback");
            flatForward = Vector3.forward;
        }

        flatForward.Normalize();

        // 水平右方向
        Vector3 flatRight = Vector3.Cross(Vector3.up, flatForward).normalized;

        // 放到相机前方 + 上移 + 左右偏移
        Vector3 panelPosition =
            cameraPosition
            + flatForward * distance
            + Vector3.up * panelVerticalOffset
            + flatRight * panelHorizontalOffset;

        Game_M.initialize.XianShi("spm_place_03_position_done");

        // World Space Canvas 和相机同向
        Quaternion panelRotation = Quaternion.LookRotation(flatForward, Vector3.up);

        panelRoot.transform.SetPositionAndRotation(panelPosition, panelRotation);

        Game_M.initialize.XianShi("spm_place_04_transform_set");

        // 如有需要，可临时强制缩放测试
        // panelRoot.transform.localScale = Vector3.one * 0.001f;
        // Game_M.initialize.XianShi("spm_place_05_scale_force");
    }
}