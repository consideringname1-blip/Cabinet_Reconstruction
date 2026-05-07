using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class RuntimeModelPoseData
{
    public bool HasWorldPose;
    public Vector3 WorldPosition = Vector3.zero;
    public Quaternion WorldRotation = Quaternion.identity;

    public bool HasArucoPose;
    public Vector3 ArucoLocalPosition = Vector3.zero;
    public Quaternion ArucoLocalRotation = Quaternion.identity;

    public bool HasResponseArucoReference;
    public Vector3 ResponseArucoReferencePosition = Vector3.zero;
    public Quaternion ResponseArucoReferenceRotation = Quaternion.identity;
}

public class RuntimeModelInstance
{
    public string ModelKey = "";
    public string TaskId = "";
    public string FbxUrl = "";
    public RuntimeModelPoseData Pose = new RuntimeModelPoseData();
}

public class RuntimeModelRecord
{
    public string ModelKey = "";
    public string TaskId = "";
    public string FbxUrl = "";
    public string LocalPath = "";
    public GameObject RootGameObject;
    public DateTime CreatedAtUtc;
    public DateTime LastTouchedAtUtc;
    public RuntimeModelPoseData Pose = new RuntimeModelPoseData();
}

[DisallowMultipleComponent]
public class RuntimeModelManager : MonoBehaviour
{
    public const string CACHE_FOLDER_NAME = "RuntimeModels";

    [SerializeField] private int maxVisibleModels = 5;
    [SerializeField] private int maxCachedModelFiles = 5;

    private static RuntimeModelManager _instance;

    private readonly List<RuntimeModelRecord> _records = new List<RuntimeModelRecord>();
    private string _cacheRootPath = "";
    private bool _hasCurrentArucoReference;
    private Vector3 _currentArucoReferencePosition = Vector3.zero;
    private Quaternion _currentArucoReferenceRotation = Quaternion.identity;

    public static RuntimeModelManager Instance
    {
        get
        {
            EnsureInstance();
            return _instance;
        }
    }

    public string CacheRootPath
    {
        get
        {
            EnsureInitialized();
            return _cacheRootPath;
        }
    }

    public string RuntimeModelCachePath
    {
        get { return CacheRootPath; }
    }

    public int MaxVisibleModels
    {
        get { return Mathf.Max(1, maxVisibleModels); }
    }

    public int MaxCachedModelFiles
    {
        get { return Mathf.Max(1, maxCachedModelFiles); }
    }

    private static void EnsureInstance()
    {
        if (_instance != null)
        {
            return;
        }

        RuntimeModelManager existing = FindObjectOfType<RuntimeModelManager>();
        if (existing != null)
        {
            _instance = existing;
            return;
        }
    }

    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
            return;
        }

        _instance = this;
        EnsureInitialized();
        ClearRuntimeStateAndCache();
    }

    private void EnsureInitialized()
    {
        if (!string.IsNullOrEmpty(_cacheRootPath))
        {
            return;
        }

        _cacheRootPath = Path.Combine(ResolveApplicationCacheRoot(), CACHE_FOLDER_NAME);
        Directory.CreateDirectory(_cacheRootPath);
    }

    private string ResolveApplicationCacheRoot()
    {
#if WINDOWS_UWP && !UNITY_EDITOR
        return Windows.Storage.ApplicationData.Current.LocalCacheFolder.Path;
#else
        return Application.persistentDataPath;
#endif
    }

    private void ClearRuntimeStateAndCache()
    {
        foreach (RuntimeModelRecord record in new List<RuntimeModelRecord>(_records))
        {
            DestroyRecordObject(record);
        }
        _records.Clear();

        ClearCacheDirectory();
    }

    public void PrepareForIncomingModel(RuntimeModelInstance instance)
    {
        if (instance == null || string.IsNullOrEmpty(instance.ModelKey))
        {
            throw new ArgumentException("Runtime model instance requires a model key.");
        }

        EnsureInitialized();
        RemoveModel(instance.ModelKey);

        while (_records.Count >= MaxVisibleModels)
        {
            RemoveOldestModel();
        }
    }

    public string CreateUniqueModelPath(string modelKey)
    {
        EnsureInitialized();
        string safeKey = SanitizeFileName(string.IsNullOrEmpty(modelKey) ? "model" : modelKey);
        string fileName = safeKey + "_" + DateTime.UtcNow.Ticks.ToString() + ".fbx";
        return Path.Combine(_cacheRootPath, fileName);
    }

    public void RegisterLoadedModel(RuntimeModelInstance instance, string localPath, GameObject rootGameObject)
    {
        if (instance == null || rootGameObject == null)
        {
            return;
        }

        PrepareForIncomingModel(instance);

        RuntimeModelRecord record = new RuntimeModelRecord
        {
            ModelKey = instance.ModelKey,
            TaskId = instance.TaskId,
            FbxUrl = instance.FbxUrl,
            LocalPath = localPath ?? "",
            RootGameObject = rootGameObject,
            CreatedAtUtc = DateTime.UtcNow,
            LastTouchedAtUtc = DateTime.UtcNow,
            Pose = instance.Pose ?? new RuntimeModelPoseData(),
        };
        _records.Add(record);
        ApplyResolvedPose(record);
        EnforceCachedFileLimit();
    }

    public bool UpdateModelPose(string taskId, RuntimeModelPoseData pose)
    {
        if (string.IsNullOrEmpty(taskId) || pose == null)
        {
            return false;
        }

        foreach (RuntimeModelRecord record in _records)
        {
            if (record == null)
            {
                continue;
            }

            if (record.TaskId == taskId || record.ModelKey == taskId)
            {
                record.Pose = pose;
                ApplyResolvedPose(record);
                return true;
            }
        }

        return false;
    }

    public void DeleteCachedFile(string localPath)
    {
        if (string.IsNullOrEmpty(localPath))
        {
            return;
        }

        try
        {
            if (File.Exists(localPath))
            {
                File.Delete(localPath);
            }
        }
        catch (Exception exc)
        {
            Debug.LogWarning("[RuntimeModelManager] Failed to delete cached model file: " + exc.Message);
        }
    }

    public int ClearLocalRuntimeModels()
    {
        EnsureInitialized();
        int removedCount = _records.Count;
        foreach (RuntimeModelRecord record in new List<RuntimeModelRecord>(_records))
        {
            DestroyRecordObject(record);
        }
        _records.Clear();
        ClearCacheDirectory();
        return removedCount;
    }

    internal void SetArucoReference(Vector3 position, Quaternion rotation)
    {
        _currentArucoReferencePosition = position;
        _currentArucoReferenceRotation = rotation;
        _hasCurrentArucoReference = true;

        foreach (RuntimeModelRecord record in _records)
        {
            if (record.Pose != null && record.Pose.HasArucoPose)
            {
                ApplyResolvedPose(record);
            }
        }
    }

    public bool TryGetCurrentArucoReference(out Vector3 position, out Quaternion rotation)
    {
        position = _currentArucoReferencePosition;
        rotation = _currentArucoReferenceRotation;
        return _hasCurrentArucoReference;
    }

    public bool TryResolveWorldPose(RuntimeModelPoseData pose, out Vector3 position, out Quaternion rotation)
    {
        position = Vector3.zero;
        rotation = Quaternion.identity;
        if (pose == null)
        {
            return false;
        }

        if (pose.HasArucoPose)
        {
            if (_hasCurrentArucoReference)
            {
                ComposeArucoWorldPose(
                    _currentArucoReferencePosition,
                    _currentArucoReferenceRotation,
                    pose,
                    out position,
                    out rotation
                );
                return true;
            }

            if (pose.HasResponseArucoReference)
            {
                ComposeArucoWorldPose(
                    pose.ResponseArucoReferencePosition,
                    pose.ResponseArucoReferenceRotation,
                    pose,
                    out position,
                    out rotation
                );
                return true;
            }
        }

        if (pose.HasWorldPose)
        {
            position = pose.WorldPosition;
            rotation = pose.WorldRotation;
            return true;
        }

        return false;
    }

    private void ApplyResolvedPose(RuntimeModelRecord record)
    {
        if (record == null || record.RootGameObject == null)
        {
            return;
        }

        if (TryResolveWorldPose(record.Pose, out Vector3 position, out Quaternion rotation))
        {
            record.RootGameObject.transform.SetPositionAndRotation(position, rotation);
            record.LastTouchedAtUtc = DateTime.UtcNow;
        }
    }

    private void ComposeArucoWorldPose(
        Vector3 arucoPosition,
        Quaternion arucoRotation,
        RuntimeModelPoseData pose,
        out Vector3 position,
        out Quaternion rotation
    )
    {
        position = arucoPosition + (arucoRotation * pose.ArucoLocalPosition);
        rotation = arucoRotation * pose.ArucoLocalRotation;
    }

    private void RemoveModel(string modelKey)
    {
        if (string.IsNullOrEmpty(modelKey))
        {
            return;
        }

        for (int i = _records.Count - 1; i >= 0; i--)
        {
            if (_records[i].ModelKey == modelKey)
            {
                RuntimeModelRecord record = _records[i];
                _records.RemoveAt(i);
                DestroyRecordObject(record);
                DeleteCachedFile(record.LocalPath);
            }
        }
    }

    private void RemoveOldestModel()
    {
        if (_records.Count == 0)
        {
            return;
        }

        int oldestIndex = 0;
        DateTime oldestTime = _records[0].CreatedAtUtc;
        for (int i = 1; i < _records.Count; i++)
        {
            if (_records[i].CreatedAtUtc < oldestTime)
            {
                oldestTime = _records[i].CreatedAtUtc;
                oldestIndex = i;
            }
        }

        RuntimeModelRecord oldest = _records[oldestIndex];
        _records.RemoveAt(oldestIndex);
        DestroyRecordObject(oldest);
        DeleteCachedFile(oldest.LocalPath);
    }

    private void DestroyRecordObject(RuntimeModelRecord record)
    {
        if (record == null || record.RootGameObject == null)
        {
            return;
        }

        Destroy(record.RootGameObject);
        record.RootGameObject = null;
    }

    private void EnforceCachedFileLimit()
    {
        EnsureInitialized();
        DirectoryInfo directory = new DirectoryInfo(_cacheRootPath);
        FileInfo[] files = directory.Exists ? directory.GetFiles("*.fbx") : new FileInfo[0];
        if (files.Length <= MaxCachedModelFiles)
        {
            return;
        }

        Array.Sort(files, (left, right) => left.CreationTimeUtc.CompareTo(right.CreationTimeUtc));
        int removeCount = files.Length - MaxCachedModelFiles;
        int removed = 0;
        for (int i = 0; i < files.Length && removed < removeCount; i++)
        {
            string path = files[i].FullName;
            if (IsActiveModelPath(path))
            {
                continue;
            }
            DeleteCachedFile(path);
            removed++;
        }
    }

    private bool IsActiveModelPath(string path)
    {
        foreach (RuntimeModelRecord record in _records)
        {
            if (!string.IsNullOrEmpty(record.LocalPath)
                && string.Equals(record.LocalPath, path, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    private void ClearCacheDirectory()
    {
        EnsureInitialized();
        try
        {
            if (!Directory.Exists(_cacheRootPath))
            {
                Directory.CreateDirectory(_cacheRootPath);
                return;
            }

            DirectoryInfo directory = new DirectoryInfo(_cacheRootPath);
            foreach (FileInfo file in directory.GetFiles())
            {
                file.Delete();
            }
        }
        catch (Exception exc)
        {
            Debug.LogWarning("[RuntimeModelManager] Failed to clear runtime model cache: " + exc.Message);
        }
    }

    private string SanitizeFileName(string raw)
    {
        char[] invalidChars = Path.GetInvalidFileNameChars();
        char[] chars = raw.ToCharArray();
        for (int i = 0; i < chars.Length; i++)
        {
            for (int j = 0; j < invalidChars.Length; j++)
            {
                if (chars[i] == invalidChars[j])
                {
                    chars[i] = '_';
                    break;
                }
            }
        }

        string safe = new string(chars).Trim();
        if (safe.Length > 80)
        {
            safe = safe.Substring(0, 80);
        }
        return string.IsNullOrEmpty(safe) ? "model" : safe;
    }
}
