using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class ModelEventDisplay : MonoBehaviour
{
    private static ModelEventDisplay _instance;
    private readonly Dictionary<string, GameObject> activeHints = new Dictionary<string, GameObject>();

    public static ModelEventDisplay Instance
    {
        get
        {
            if (_instance != null)
            {
                return _instance;
            }

            _instance = FindObjectOfType<ModelEventDisplay>();
            if (_instance == null)
            {
                GameObject displayObject = new GameObject("ModelEventDisplay");
                _instance = displayObject.AddComponent<ModelEventDisplay>();
            }
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
        if (identity == null)
        {
            return;
        }

        string key = identity.ModelKey;
        if (activeHints.ContainsKey(key))
        {
            CloseHint(key);
            ToggleObjectEvidence(identity);
            return;
        }

        bool evidenceHandled = ToggleObjectEvidence(identity);
        if (evidenceHandled)
        {
            return;
        }

        RuntimeModelRecord record = null;
        RuntimeModelManager manager = RuntimeModelManager.Instance;
        bool found = manager != null
            && !string.IsNullOrEmpty(identity.ModelKey)
            && manager.TryGetLoadedRecord(identity.ModelKey, out record);
        if (found)
        {
            ShowFrontMessage("model_event_no_evidence");
            return;
        }

        if (!evidenceHandled)
        {
            ShowFrontMessage("model_event_no_local_hint");
        }
    }

    private bool ToggleObjectEvidence(RuntimeModelEventIdentity identity)
    {
        if (identity == null)
        {
            return false;
        }

        ObjectEvidenceDisplay evidenceDisplay = ObjectEvidenceDisplay.Instance;
        return evidenceDisplay != null
            && !string.IsNullOrEmpty(identity.DisplayObjectId)
            && evidenceDisplay.ToggleEvidenceForModel(identity.DisplayObjectId);
    }

    public void ShowForModel(RuntimeModelInstance instance, string message)
    {
        if (instance == null || instance.SpatialBox == null || !instance.SpatialBox.IsReady)
        {
            return;
        }

        string key = instance.ModelKey;
        if (string.IsNullOrEmpty(key))
        {
            return;
        }

        GameObject root;
        RuntimeSpatialBoxDisplay display;
        if (!activeHints.TryGetValue(key, out root) || root == null)
        {
            root = new GameObject("RuntimeSpatialHint_" + key);
            display = root.AddComponent<RuntimeSpatialBoxDisplay>();
            activeHints[key] = root;
        }
        else
        {
            display = root.GetComponent<RuntimeSpatialBoxDisplay>();
            if (display == null)
            {
                display = root.AddComponent<RuntimeSpatialBoxDisplay>();
            }
        }

        display.Configure(instance.SpatialBox, message);
    }

    public void UpdateProgressForModel(RuntimeModelInstance instance, string message)
    {
        if (instance == null)
        {
            return;
        }

        string key = instance.ModelKey;
        if (string.IsNullOrEmpty(key))
        {
            return;
        }

        GameObject root;
        if (!activeHints.TryGetValue(key, out root) || root == null)
        {
            ShowForModel(instance, message);
            return;
        }

        RuntimeSpatialBoxDisplay display = root.GetComponent<RuntimeSpatialBoxDisplay>();
        if (display != null)
        {
            display.UpdateMessage(message);
        }
    }

    public void CloseForModel(RuntimeModelInstance instance)
    {
        if (instance == null)
        {
            return;
        }

        CloseHint(instance.ModelKey);
    }

    public void CloseAllAndClearLocalCache()
    {
        foreach (GameObject hint in new List<GameObject>(activeHints.Values))
        {
            if (hint != null)
            {
                Destroy(hint);
            }
        }
        activeHints.Clear();
    }

    public void DeleteServerEventsForTaskIds(IEnumerable<string> taskIds)
    {
        if (taskIds == null)
        {
            return;
        }

        foreach (string taskId in taskIds)
        {
            CloseHint(taskId);
        }
    }

    private void CloseHint(string key)
    {
        if (string.IsNullOrEmpty(key))
        {
            return;
        }

        GameObject hint;
        if (activeHints.TryGetValue(key, out hint))
        {
            if (hint != null)
            {
                Destroy(hint);
            }
            activeHints.Remove(key);
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
