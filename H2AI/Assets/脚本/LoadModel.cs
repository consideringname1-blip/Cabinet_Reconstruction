using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using System.Collections;
using System.Collections.Generic;
using TriLibCore;
using UnityEngine;
/// <summary>
/// 加载
/// </summary>
public class LoadModel : MonoBehaviour
{

    public static LoadModel initialize;


    public bool hasServerPose = false;
    public Vector3 serverObjectPosition = Vector3.zero;
    public Quaternion serverObjectRotation = Quaternion.identity;

    /// <summary>
    /// Loads the "Models/TriLibSample.obj" Model using the given AssetLoaderOptions.
    /// </summary>
    /// <remarks>
    /// You can create the AssetLoaderOptions by right clicking on the Assets Explorer and selecting "TriLib->Create->AssetLoaderOptions->Pre-Built AssetLoaderOptions".
    /// </remarks>
    private void Start()
    {
        initialize = this;
        assetLoaderOptions = AssetLoader.CreateDefaultLoaderOptions();

    }

    public void Update()
    {
        if (Input.GetKeyDown(KeyCode.M))
        {
            YanChiJiaZai();
        }
    }
    AssetLoaderOptions assetLoaderOptions;
    /// <summary>
    /// 延迟加载
    /// </summary>
    public void YanChiJiaZai()
    {
        if (xiaZaiModel!=null)
        {
            xiaZaiModel.gameObject.SetActive(false);
        }
        Game_M.initialize.XianShi("zaiRu");
        CancelInvoke();
        Invoke("OnJiaZai", 1);
    }

    /// <summary>
    /// 加载
    /// </summary>
     void OnJiaZai()
    {
   
        string ModelPath = Application.streamingAssetsPath + "/model.fbx";

#if !UNITY_EDITOR
          ModelPath = Windows.Storage.ApplicationData.Current.RoamingFolder.Path + "/model.fbx";
#endif

        AssetLoader.LoadModelFromFile(ModelPath, OnLoad, OnMaterialsLoad, OnProgress, OnError, null, assetLoaderOptions);
    }
    /// <summary>
    /// Called when any error occurs.
    /// </summary>
    /// <param name="obj">The contextualized error, containing the original exception and the context passed to the method where the error was thrown.</param>
    private void OnError(IContextualizedError obj)
    {
        Debug.LogError($"An error occurred while loading your Model: {obj.GetInnerException()}");
    }

    /// <summary>
    /// Called when the Model loading progress changes.
    /// </summary>
    /// <param name="assetLoaderContext">The context used to load the Model.</param>
    /// <param name="progress">The loading progress.</param>
    private void OnProgress(AssetLoaderContext assetLoaderContext, float progress)
    {
        Debug.Log($"Loading Model. Progress: {progress:P}");
        Game_M.initialize.XianShi(progress.ToString());
    }
    public GameObject xiaZaiModel;
    /// <summary>
    /// Called when the Model (including Textures and Materials) has been fully loaded, or after any error occurs.
    /// </summary>
    /// <remarks>The loaded GameObject is available on the assetLoaderContext.RootGameObject field.</remarks>
    /// <param name="assetLoaderContext">The context used to load the Model.</param>
    private void OnMaterialsLoad(AssetLoaderContext assetLoaderContext)
    {
        Debug.Log("Materials loaded. Model fully loaded.");

        GameObject game = assetLoaderContext.RootGameObject;
        game.SetActive(true);

        // 1. 默认值：放在原点，朝向世界Z轴
        Vector3 targetPos = Vector3.zero;
        Quaternion targetRot = Quaternion.identity;

        // 2. 优先使用服务器返回的世界位姿
        if (ShuJuQingQiu.initialize != null && ShuJuQingQiu.initialize.hasServerPose)
        {
            targetPos = ShuJuQingQiu.initialize.serverObjectPosition;
            targetRot = ShuJuQingQiu.initialize.serverObjectRotation;

            Debug.Log($"[POSE] Use server pose: pos={targetPos}, rot={targetRot}");
        }
        else
        {
            // 3. 没有服务器位姿时的后备逻辑：放到相机前方 2 米
            Camera cam = Camera.main;
            if (cam != null)
            {
                Vector3 forwardFlat = new Vector3(cam.transform.forward.x, 0f, cam.transform.forward.z).normalized;
                if (forwardFlat.sqrMagnitude < 1e-4f)
                {
                    forwardFlat = cam.transform.forward.normalized;
                }

                targetPos = cam.transform.position + forwardFlat * 2f;
                // 朝向相机看向的方向（水平向前）
                targetRot = Quaternion.LookRotation(forwardFlat, Vector3.up);

                Debug.Log($"[POSE] Fallback pose: pos={targetPos}, rot={targetRot}");
            }
            else
            {
                Debug.LogWarning("[POSE] Camera.main 未找到，使用默认 (0,0,0)+identity。");
            }
        }

        // 4. 应用位姿
        game.transform.SetPositionAndRotation(targetPos, targetRot);

        // 5. 后续逻辑保持不变
        AddGameObjectCollider(game);
        game.gameObject.AddComponent<ObjectManipulator>();
        game.gameObject.AddComponent<NearInteractionGrabbable>();
        Game_M.initialize.GuanBi();
        xiaZaiModel = game;
    }

    /// <summary>
    /// 添加碰撞体
    /// </summary>
    /// <param name="gameObject"></param>  
    public static void AddGameObjectCollider(GameObject gameObject)
    {
        Vector3 pos = gameObject.transform.localPosition;
        Quaternion qt = gameObject.transform.localRotation;
        Vector3 ls = gameObject.transform.localScale;

        gameObject.transform.position = Vector3.zero;
        gameObject.transform.eulerAngles = Vector3.zero;
        gameObject.transform.localScale = Vector3.one;
        //获取物体的最小包围盒
        Bounds itemBound = GetLocalBounds(gameObject);

        gameObject.transform.localPosition = pos;
        gameObject.transform.localRotation = qt;
        gameObject.transform.localScale = ls;
        //parent = null;
        if (!gameObject.GetComponent<Collider>())
            gameObject.AddComponent<BoxCollider>();
        if (gameObject.GetComponent<BoxCollider>())
        {
            gameObject.GetComponent<BoxCollider>().size = itemBound.size;

            gameObject.GetComponent<BoxCollider>().center = itemBound.center;
        }
    }

    /// <summary>
    /// 获得对象的最小包围盒
    /// </summary>
    public static Bounds GetLocalBounds(GameObject target)
    {
        Renderer[] mfs = target.GetComponentsInChildren<Renderer>();
        Bounds bounds = new Bounds();
        if (mfs.Length != 0)
        {
            bounds = mfs[0].bounds;
            foreach (Renderer mf in mfs)
            {
                bounds.Encapsulate(mf.bounds);
            }
        }
        return bounds;
    }
    /// <summary>
    /// Called when the Model Meshes and hierarchy are loaded.
    /// </summary>
    /// <remarks>The loaded GameObject is available on the assetLoaderContext.RootGameObject field.</remarks>
    /// <param name="assetLoaderContext">The context used to load the Model.</param>
    private void OnLoad(AssetLoaderContext assetLoaderContext)
    {
        Debug.Log("Model loaded. Loading materials.");
    }
}
