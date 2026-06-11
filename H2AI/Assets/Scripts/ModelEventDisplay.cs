using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class ModelEventDisplay : MonoBehaviour
{
    private static ModelEventDisplay _instance;
    private GameObject activePopup;

    public static ModelEventDisplay Instance
    {
        get
        {
            if (_instance != null)
            {
                return _instance;
            }

            _instance = FindObjectOfType<ModelEventDisplay>();
            return _instance;
        }
    }

    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(this);
            return;
        }

        _instance = this;
    }

    private void OnDestroy()
    {
        if (_instance == this)
        {
            _instance = null;
        }

        CloseAllAndClearLocalCache();
    }

    public void ToggleForModel(RuntimeModelEventIdentity identity)
    {
        if (activePopup != null)
        {
            ClosePopup();
            return;
        }

        ShowFrontMessage("model_event_disabled");
    }

    public void CloseAllAndClearLocalCache()
    {
        ClosePopup();
    }

    public void DeleteServerEventsForTaskIds(IEnumerable<string> taskIds)
    {
        ClosePopup();
    }

    private void ClosePopup()
    {
        if (activePopup != null)
        {
            Destroy(activePopup);
            activePopup = null;
        }
    }

    private void ShowFrontMessage(string message)
    {
        if (Game_M.initialize != null)
        {
            Game_M.initialize.XianShiForSeconds(message);
        }
    }
}
