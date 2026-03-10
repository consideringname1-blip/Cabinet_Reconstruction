using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;

public class SelectionButtonsUI : MonoBehaviour
{
    [SerializeField] private SelectionBoxController selectionBoxController;

    public event Action ConfirmClicked;
    public event Action CancelClicked;

    private void Awake()
    {
        Game_M.initialize.XianShi("sbu_awake_01_enter");

        if (selectionBoxController == null)
        {
            Game_M.initialize.XianShi("sbu_ERR_awake_selectionBoxController_null");
        }
        else
        {
            Game_M.initialize.XianShi("sbu_awake_02_controller_ok");
        }
    }

    private void OnEnable()
    {
        Game_M.initialize.XianShi("sbu_onenable_01");
    }

    private void OnDisable()
    {
        Game_M.initialize.XianShi("sbu_ondisable_01");
    }

    public void OnConfirmClicked()
    {
        Game_M.initialize.XianShi("sbu_confirm_01_enter");

        if (selectionBoxController == null)
        {
            Game_M.initialize.XianShi("sbu_ERR_confirm_controller_null");

            if (ConfirmClicked != null)
            {
                Game_M.initialize.XianShi("sbu_confirm_02_invoke_noctrl");
                ConfirmClicked.Invoke();
            }
            else
            {
                Game_M.initialize.XianShi("sbu_ERR_confirm_event_null_noctrl");
            }

            return;
        }

        Game_M.initialize.XianShi("sbu_confirm_03_controller_ok");

        if (ConfirmClicked != null)
        {
            Game_M.initialize.XianShi("sbu_confirm_04_invoke");
            ConfirmClicked.Invoke();
            Game_M.initialize.XianShi("sbu_confirm_05_after_invoke");
        }
        else
        {
            Game_M.initialize.XianShi("sbu_ERR_confirm_event_null");
        }
    }

    public void OnCancelClicked()
    {
        Game_M.initialize.XianShi("sbu_cancel_01_enter");

        if (CancelClicked != null)
        {
            Game_M.initialize.XianShi("sbu_cancel_02_invoke");
            CancelClicked.Invoke();
            Game_M.initialize.XianShi("sbu_cancel_03_after_invoke");
        }
        else
        {
            Game_M.initialize.XianShi("sbu_ERR_cancel_event_null");
        }
    }
}