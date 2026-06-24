using System.IO;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using TriLibCore;
using UnityEngine;

public class LoadModel : MonoBehaviour
{
    public static LoadModel initialize;

    public event System.Action<RuntimeModelInstance, bool> RuntimeModelLoadCompleted;

    private AssetLoaderOptions _assetLoaderOptions;
    private bool _isLoading;
    private RuntimeModelInstance _pendingInstance;
    private string _pendingLocalPath = "";

    public static LoadModel Instance
    {
        get
        {
            if (initialize != null)
            {
                return initialize;
            }

            GameObject loaderObject = new GameObject("LoadModel");
            return loaderObject.AddComponent<LoadModel>();
        }
    }

    public bool IsLoading
    {
        get { return _isLoading; }
    }

    private void Awake()
    {
        if (initialize != null && initialize != this)
        {
            Destroy(this);
            return;
        }

        initialize = this;
        _assetLoaderOptions = AssetLoader.CreateDefaultLoaderOptions();
    }

    private void OnDestroy()
    {
        if (initialize == this)
        {
            initialize = null;
        }
    }

    public bool LoadRuntimeModel(RuntimeModelInstance instance, string localPath)
    {
        if (_isLoading)
        {
            Debug.LogWarning("[LoadModel] A model is already loading.");
            ShowFrontMessage("load_ERR_busy");
            return false;
        }

        if (instance == null || string.IsNullOrEmpty(instance.ModelKey))
        {
            Debug.LogError("[LoadModel] Runtime model instance is missing.");
            ShowFrontMessage("load_ERR_missing_model_instance");
            return false;
        }

        if (string.IsNullOrEmpty(localPath) || !File.Exists(localPath))
        {
            Debug.LogError("[LoadModel] Model file not found: " + localPath);
            ShowFrontMessage("load_ERR_missing_file");
            return false;
        }

        _pendingInstance = instance;
        _pendingLocalPath = localPath;
        _isLoading = true;

        ShowFrontMessage("zaiRu");
        AssetLoader.LoadModelFromFile(
            localPath,
            OnLoad,
            OnMaterialsLoad,
            OnProgress,
            OnError,
            null,
            _assetLoaderOptions
        );
        return true;
    }

    private void OnError(IContextualizedError obj)
    {
        RuntimeModelInstance failedInstance = _pendingInstance;
        _isLoading = false;
        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager != null)
        {
            manager.DeleteCachedFile(_pendingLocalPath);
        }
        ClearPendingModel();

        Debug.LogError("An error occurred while loading your Model: " + obj.GetInnerException());
        UpdateSpatialHint(failedInstance, "failed");
        ShowFrontMessage("load_ERR_failed");
        NotifyRuntimeModelLoadCompleted(failedInstance, false);
    }

    private void OnProgress(AssetLoaderContext assetLoaderContext, float progress)
    {
        string progressText = progress.ToString("P0");
        Debug.Log("Loading Model. Progress: " + progress.ToString("P"));
        UpdateSpatialHint(_pendingInstance, progressText);
        ShowFrontMessage(progressText);
    }

    private void OnMaterialsLoad(AssetLoaderContext assetLoaderContext)
    {
        RuntimeModelInstance loadedInstance = _pendingInstance;
        _isLoading = false;
        Debug.Log("Materials loaded. Model fully loaded.");

        GameObject game = assetLoaderContext.RootGameObject;
        game.name = "RuntimeModel_" + _pendingInstance.ModelKey;
        game.SetActive(true);

        RuntimeModelManager manager = RuntimeModelManager.Instance;
        if (manager != null && manager.TryResolveWorldPose(
            _pendingInstance.Pose,
            out Vector3 targetPosition,
            out Quaternion targetRotation
        ))
        {
            game.transform.SetPositionAndRotation(targetPosition, targetRotation);
            Debug.Log("[LoadModel] Use runtime pose: pos=" + targetPosition + ", rot=" + targetRotation);
        }
        else
        {
            ApplyFallbackPose(game);
        }

        AddGameObjectCollider(game);
        DisableRuntimeModelManipulation(game);

        if (manager == null)
        {
            Debug.LogError("[RuntimeModelManager] Missing RuntimeModelManager component on scene Scripts object.");
            ShowFrontMessage("runtime_model_mgr_missing");
            Destroy(game);
            ClearPendingModel();
            NotifyRuntimeModelLoadCompleted(loadedInstance, false);
            return;
        }

        manager.RegisterLoadedModel(_pendingInstance, _pendingLocalPath, game);
        HideSpatialHint(_pendingInstance);

        ShowFrontMessageForSeconds("download_completed", 3f);
        ClearPendingModel();
        NotifyRuntimeModelLoadCompleted(loadedInstance, true);
    }

    private void OnLoad(AssetLoaderContext assetLoaderContext)
    {
        Debug.Log("Model loaded. Loading materials.");
    }

    private void ApplyFallbackPose(GameObject game)
    {
        Vector3 targetPosition = Vector3.zero;
        Quaternion targetRotation = Quaternion.identity;

        Camera cam = Camera.main;
        if (cam != null)
        {
            Vector3 forwardFlat = new Vector3(cam.transform.forward.x, 0f, cam.transform.forward.z).normalized;
            if (forwardFlat.sqrMagnitude < 1e-4f)
            {
                forwardFlat = cam.transform.forward.normalized;
            }

            targetPosition = cam.transform.position + forwardFlat * 2f;
            targetRotation = Quaternion.LookRotation(forwardFlat, Vector3.up);
        }
        else
        {
            Debug.LogWarning("[LoadModel] Camera.main was not found. Using identity fallback pose.");
        }

        game.transform.SetPositionAndRotation(targetPosition, targetRotation);
    }

    private void ClearPendingModel()
    {
        _pendingInstance = null;
        _pendingLocalPath = "";
    }

    private void ShowFrontMessage(string message)
    {
        if (Game_M.initialize != null)
        {
            Game_M.initialize.XianShi(message);
        }
    }

    private void ShowFrontMessageForSeconds(string message, float seconds)
    {
        if (Game_M.initialize != null)
        {
            Game_M.initialize.XianShiForSeconds(message, seconds);
        }
    }

    private void UpdateSpatialHint(RuntimeModelInstance instance, string message)
    {
        ModelEventDisplay display = ModelEventDisplay.Instance;
        if (display != null)
        {
            display.UpdateProgressForModel(instance, message);
        }
    }

    private void HideSpatialHint(RuntimeModelInstance instance)
    {
        ModelEventDisplay display = ModelEventDisplay.Instance;
        if (display != null)
        {
            display.CloseForModel(instance);
        }
    }

    private void NotifyRuntimeModelLoadCompleted(RuntimeModelInstance instance, bool success)
    {
        System.Action<RuntimeModelInstance, bool> handler = RuntimeModelLoadCompleted;
        if (handler != null)
        {
            handler.Invoke(instance, success);
        }
    }


    private static void DisableRuntimeModelManipulation(GameObject gameObject)
    {
        foreach (ObjectManipulator manipulator in gameObject.GetComponentsInChildren<ObjectManipulator>(true))
        {
            Destroy(manipulator);
        }

        foreach (NearInteractionGrabbable grabbable in gameObject.GetComponentsInChildren<NearInteractionGrabbable>(true))
        {
            Destroy(grabbable);
        }
    }

    private static void AddGameObjectCollider(GameObject gameObject)
    {
        Vector3 pos = gameObject.transform.localPosition;
        Quaternion qt = gameObject.transform.localRotation;
        Vector3 ls = gameObject.transform.localScale;

        gameObject.transform.position = Vector3.zero;
        gameObject.transform.eulerAngles = Vector3.zero;
        gameObject.transform.localScale = Vector3.one;

        Bounds itemBound = GetLocalBounds(gameObject);

        gameObject.transform.localPosition = pos;
        gameObject.transform.localRotation = qt;
        gameObject.transform.localScale = ls;

        if (!gameObject.GetComponent<Collider>())
        {
            gameObject.AddComponent<BoxCollider>();
        }

        BoxCollider boxCollider = gameObject.GetComponent<BoxCollider>();
        if (boxCollider != null)
        {
            boxCollider.size = itemBound.size;
            boxCollider.center = itemBound.center;
        }
    }

    private static Bounds GetLocalBounds(GameObject target)
    {
        Renderer[] renderers = target.GetComponentsInChildren<Renderer>();
        Bounds bounds = new Bounds();
        if (renderers.Length != 0)
        {
            bounds = renderers[0].bounds;
            foreach (Renderer renderer in renderers)
            {
                bounds.Encapsulate(renderer.bounds);
            }
        }
        return bounds;
    }
}
