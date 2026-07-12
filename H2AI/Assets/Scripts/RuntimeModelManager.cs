using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class RuntimeModelPoseData
{
    public bool HasHololensPose;
    public Vector3 HololensPosition = Vector3.zero;
    public Quaternion HololensRotation = Quaternion.identity;
}


public class RuntimeSpatialBoxData
{
    public bool IsReady;
    public string Status = "";
    public string CoordinateSpace = "unity_world";
    public Vector3 AabbMinWorld = Vector3.zero;
    public Vector3 AabbMaxWorld = Vector3.zero;

    public Vector3 CenterWorld
    {
        get { return (AabbMinWorld + AabbMaxWorld) * 0.5f; }
    }

    public Vector3 SizeWorld
    {
        get
        {
            Vector3 size = AabbMaxWorld - AabbMinWorld;
            return new Vector3(Mathf.Abs(size.x), Mathf.Abs(size.y), Mathf.Abs(size.z));
        }
    }
}

public class RuntimeModelInstance
{
    public string ModelKey = "";
    public string TaskId = "";
    public string FbxUrl = "";
    public string DisplayObjectId = "";
    public long ModelRevision = -1;
    public bool IsEvidenceOverlay;
    public RuntimeModelPoseData Pose = new RuntimeModelPoseData();
    public RuntimeSpatialBoxData SpatialBox;
}

public class RuntimeModelRecord
{
    public string ModelKey = "";
    public string TaskId = "";
    public string FbxUrl = "";
    public string DisplayObjectId = "";
    public long ModelRevision = -1;
    public long HololensPoseRevision = -1;
    public long TrackingPoseRevision = -1;
    public long LastAppliedModeEpoch = -1;
    public string CoordinateEpoch = "";
    public string AppliedPoseSource = "";
    public string LocalPath = "";
    public bool IsEvidenceOverlay;
    public GameObject RootGameObject;
    public DateTime CreatedAtUtc;
    public DateTime LastTouchedAtUtc;
    public RuntimeModelPoseData Pose = new RuntimeModelPoseData();
    public RuntimeSpatialBoxData SpatialBox;
}

[DisallowMultipleComponent]
public class RuntimeModelManager : MonoBehaviour
{
    public const string CACHE_FOLDER_NAME = "RuntimeModels";

    [SerializeField] private int maxVisibleModels = 5;
    [SerializeField] private int maxCachedModelFiles = 5;

    private static RuntimeModelManager _instance;

    private readonly List<RuntimeModelRecord> _records = new List<RuntimeModelRecord>();
    private readonly HashSet<string> _protectedCachePaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    private string _cacheRootPath = "";
    private bool _runtimeModelsVisible = true;

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

    public void PrepareForIncomingModel(RuntimeModelInstance instance)
    {
        if (instance == null
            || string.IsNullOrEmpty(instance.ModelKey)
            || string.IsNullOrEmpty(instance.TaskId)
            || string.IsNullOrEmpty(instance.DisplayObjectId)
            || instance.ModelRevision <= 0
            || string.IsNullOrEmpty(instance.FbxUrl)
            || instance.Pose == null
            || !instance.Pose.HasHololensPose)
        {
            throw new ArgumentException("Runtime model instance does not satisfy the canonical model contract.");
        }

        EnsureInitialized();
        // Keep the current revision alive while the replacement downloads/imports.
        // Superseded records are removed only after RegisterLoadedModel receives a
        // successfully staged GameObject.
    }

    public string CreateStableModelPath(RuntimeModelInstance instance)
    {
        if (instance == null)
        {
            throw new ArgumentNullException("instance");
        }
        EnsureInitialized();

        PrepareForIncomingModel(instance);
        string prefix = instance.IsEvidenceOverlay ? "evidence_display_" : "display_";
        string fileName = prefix + GetStableDisplayCacheKey(instance.DisplayObjectId)
            + "_revision_" + instance.ModelRevision.ToString() + ".fbx";
        return Path.Combine(_cacheRootPath, fileName);
    }

    public bool TryGetCachedModelPath(RuntimeModelInstance instance, out string localPath)
    {
        localPath = "";
        if (instance == null)
        {
            return false;
        }
        string candidate = CreateStableModelPath(instance);
        try
        {
            FileInfo file = new FileInfo(candidate);
            if (!file.Exists || file.Length <= 0)
            {
                return false;
            }
            localPath = candidate;
            return true;
        }
        catch (Exception exc)
        {
            Debug.LogWarning("[RuntimeModelManager] Failed to inspect cached model: " + exc.Message);
            return false;
        }
    }

    public void ProtectCachedModelPath(string localPath)
    {
        if (!string.IsNullOrEmpty(localPath))
        {
            _protectedCachePaths.Add(localPath);
        }
    }

    public void UnprotectCachedModelPath(string localPath)
    {
        if (!string.IsNullOrEmpty(localPath))
        {
            _protectedCachePaths.Remove(localPath);
        }
    }

    public void RegisterLoadedModel(RuntimeModelInstance instance, string localPath, GameObject rootGameObject)
    {
        if (instance == null || rootGameObject == null)
        {
            return;
        }

        PrepareForIncomingModel(instance);

        bool replacementWasVisible = _runtimeModelsVisible;
        RuntimeModelRecord replacedDisplayRecord = FindDisplayObjectRecord(instance.DisplayObjectId);
        if (replacedDisplayRecord != null && replacedDisplayRecord.RootGameObject != null)
        {
            replacementWasVisible = replacementWasVisible && replacedDisplayRecord.RootGameObject.activeSelf;
        }

        RuntimeModelRecord record = new RuntimeModelRecord
        {
            ModelKey = instance.ModelKey,
            TaskId = instance.TaskId,
            FbxUrl = instance.FbxUrl,
            DisplayObjectId = instance.DisplayObjectId,
            ModelRevision = instance.ModelRevision,
            LocalPath = localPath ?? "",
            IsEvidenceOverlay = instance.IsEvidenceOverlay,
            RootGameObject = rootGameObject,
            CreatedAtUtc = DateTime.UtcNow,
            LastTouchedAtUtc = DateTime.UtcNow,
            Pose = instance.Pose ?? new RuntimeModelPoseData(),
            SpatialBox = instance.SpatialBox,
        };
        _records.Add(record);
        if (!record.IsEvidenceOverlay)
        {
            AttachEventIdentity(record);
        }
        ApplyResolvedPose(record);
        rootGameObject.SetActive(replacementWasVisible);

        RemoveSupersededRecordsAfterSuccessfulStage(record);
        while (CountDisplayObjectBundles() > MaxVisibleModels)
        {
            if (!RemoveOldestDisplayObjectBundle())
            {
                break;
            }
        }
        ModelEventDisplay display = ModelEventDisplay.Instance;
        if (display != null)
        {
            display.CloseForModel(instance);
        }
        DeleteSupersededStableCacheFiles(record);
        EnforceCachedFileLimit();
    }

    public bool HasMatchingModel(RuntimeModelInstance instance)
    {
        if (instance == null)
        {
            return false;
        }

        foreach (RuntimeModelRecord record in _records)
        {
            if (MatchesRuntimeModelIdentity(record, instance))
            {
                return true;
            }
        }

        return false;
    }

    public bool TryGetLoadedRecord(string modelKey, out RuntimeModelRecord matchedRecord)
    {
        matchedRecord = null;
        if (string.IsNullOrEmpty(modelKey))
        {
            return false;
        }

        foreach (RuntimeModelRecord record in _records)
        {
            if (record == null)
            {
                continue;
            }

            if (record.ModelKey == modelKey)
            {
                matchedRecord = record;
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Applies a server pose to the currently loaded revision of one logical display object.
    /// Revisions are tracked independently for HoloLens-confirmed and realtime-tracking poses,
    /// because switching to history mode intentionally changes pose source.
    /// </summary>
    public bool UpdateDisplayObjectPose(
        string displayObjectId,
        long modelRevision,
        string poseSource,
        long sourcePoseRevision,
        long modeEpoch,
        string coordinateEpoch,
        RuntimeModelPoseData pose,
        out string rejectionReason)
    {
        rejectionReason = "";
        if (string.IsNullOrEmpty(displayObjectId))
        {
            rejectionReason = "display_object_id_missing";
            return false;
        }
        if (pose == null || !pose.HasHololensPose)
        {
            rejectionReason = "hololens_pose_missing";
            return false;
        }

        string normalizedSource = NormalizePoseSource(poseSource);
        if (string.IsNullOrEmpty(normalizedSource))
        {
            rejectionReason = "pose_source_invalid";
            return false;
        }

        RuntimeModelRecord record = null;
        foreach (RuntimeModelRecord candidate in _records)
        {
            if (MatchesDisplayObject(candidate, displayObjectId))
            {
                record = candidate;
                break;
            }
        }
        if (record == null || record.RootGameObject == null)
        {
            rejectionReason = "display_object_model_not_loaded";
            return false;
        }

        if (modelRevision <= 0 || record.ModelRevision != modelRevision)
        {
            rejectionReason = "model_revision_mismatch";
            return false;
        }
        if (modeEpoch < 0)
        {
            rejectionReason = "mode_epoch_missing";
            return false;
        }
        if (record.LastAppliedModeEpoch >= 0 && modeEpoch < record.LastAppliedModeEpoch)
        {
            rejectionReason = "mode_epoch_stale";
            return false;
        }
        if (string.IsNullOrEmpty(coordinateEpoch))
        {
            rejectionReason = "coordinate_epoch_missing";
            return false;
        }

        long currentSourceRevision = normalizedSource == "tracking"
            ? record.TrackingPoseRevision
            : record.HololensPoseRevision;
        if (sourcePoseRevision <= 0)
        {
            rejectionReason = "pose_revision_missing";
            return false;
        }
        if (currentSourceRevision >= 0 && sourcePoseRevision < currentSourceRevision)
        {
            rejectionReason = "pose_revision_stale";
            return false;
        }

        bool modeAdvanced = modeEpoch >= 0 && modeEpoch > record.LastAppliedModeEpoch;
        bool coordinateAdvanced = !string.Equals(record.CoordinateEpoch, coordinateEpoch, StringComparison.Ordinal);
        bool sourceChanged = !string.Equals(record.AppliedPoseSource, normalizedSource, StringComparison.Ordinal);
        if (sourcePoseRevision >= 0
            && sourcePoseRevision == currentSourceRevision
            && !modeAdvanced
            && !coordinateAdvanced
            && !sourceChanged)
        {
            rejectionReason = "pose_revision_duplicate";
            return false;
        }

        if (normalizedSource == "tracking")
        {
            record.TrackingPoseRevision = Math.Max(record.TrackingPoseRevision, sourcePoseRevision);
        }
        else
        {
            record.HololensPoseRevision = Math.Max(record.HololensPoseRevision, sourcePoseRevision);
        }
        record.LastAppliedModeEpoch = Math.Max(record.LastAppliedModeEpoch, modeEpoch);
        record.CoordinateEpoch = coordinateEpoch;
        record.AppliedPoseSource = normalizedSource;
        record.Pose = pose;
        ApplyResolvedPose(record);
        return true;
    }

    public bool TryGetLoadedRecordByDisplayObjectId(string displayObjectId, out RuntimeModelRecord matchedRecord)
    {
        matchedRecord = null;
        if (string.IsNullOrEmpty(displayObjectId))
        {
            return false;
        }
        foreach (RuntimeModelRecord record in _records)
        {
            if (MatchesDisplayObject(record, displayObjectId))
            {
                matchedRecord = record;
                return true;
            }
        }
        return false;
    }

    private static string NormalizePoseSource(string poseSource)
    {
        string normalized = (poseSource ?? "").Trim().ToLowerInvariant();
        if (normalized == "tracking")
        {
            return "tracking";
        }
        if (normalized == "hololens")
        {
            return "hololens";
        }
        return "";
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

    public int HideAllRuntimeModels()
    {
        _runtimeModelsVisible = false;
        int hiddenCount = 0;
        foreach (RuntimeModelRecord record in _records)
        {
            if (record == null || record.RootGameObject == null)
            {
                continue;
            }
            record.RootGameObject.SetActive(false);
            hiddenCount++;
        }
        return hiddenCount;
    }

    public int ShowAllRuntimeModels()
    {
        _runtimeModelsVisible = true;
        int shownCount = 0;
        foreach (RuntimeModelRecord record in _records)
        {
            if (record == null || record.RootGameObject == null)
            {
                continue;
            }
            record.RootGameObject.SetActive(true);
            shownCount++;
        }
        return shownCount;
    }

    public bool TryResolveWorldPose(RuntimeModelPoseData pose, out Vector3 position, out Quaternion rotation)
    {
        position = Vector3.zero;
        rotation = Quaternion.identity;
        if (pose == null)
        {
            return false;
        }

        if (pose.HasHololensPose)
        {
            position = pose.HololensPosition;
            rotation = pose.HololensRotation;
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

    private RuntimeModelRecord FindDisplayObjectRecord(string displayObjectId)
    {
        if (string.IsNullOrEmpty(displayObjectId))
        {
            return null;
        }
        foreach (RuntimeModelRecord record in _records)
        {
            if (MatchesDisplayObject(record, displayObjectId))
            {
                return record;
            }
        }
        return null;
    }

    private void RemoveSupersededRecordsAfterSuccessfulStage(RuntimeModelRecord stagedRecord)
    {
        if (stagedRecord == null)
        {
            return;
        }
        for (int i = _records.Count - 1; i >= 0; i--)
        {
            RuntimeModelRecord candidate = _records[i];
            if (candidate == null || ReferenceEquals(candidate, stagedRecord))
            {
                continue;
            }

            bool superseded;
            if (stagedRecord.IsEvidenceOverlay)
            {
                superseded = candidate.IsEvidenceOverlay
                    && !string.IsNullOrEmpty(stagedRecord.ModelKey)
                    && string.Equals(candidate.ModelKey, stagedRecord.ModelKey, StringComparison.Ordinal);
            }
            else if (!string.IsNullOrEmpty(stagedRecord.DisplayObjectId))
            {
                superseded = MatchesDisplayObject(candidate, stagedRecord.DisplayObjectId);
            }
            else
            {
                superseded = false;
            }
            if (!superseded)
            {
                continue;
            }

            _records.RemoveAt(i);
            DestroyRecordObject(candidate);
            DeleteCachedFile(candidate.LocalPath);
        }
    }

    private string GetDisplayObjectBundleKey(RuntimeModelRecord record)
    {
        if (record == null)
        {
            return "";
        }
        return string.IsNullOrEmpty(record.DisplayObjectId)
            ? ""
            : "display:" + record.DisplayObjectId;
    }

    private int CountDisplayObjectBundles()
    {
        HashSet<string> bundleKeys = new HashSet<string>(StringComparer.Ordinal);
        foreach (RuntimeModelRecord record in _records)
        {
            if (record == null || record.IsEvidenceOverlay)
            {
                continue;
            }
            string bundleKey = GetDisplayObjectBundleKey(record);
            if (!string.IsNullOrEmpty(bundleKey))
            {
                bundleKeys.Add(bundleKey);
            }
        }
        return bundleKeys.Count;
    }

    private bool RemoveOldestDisplayObjectBundle()
    {
        string oldestBundleKey = "";
        DateTime oldestTime = DateTime.MaxValue;
        foreach (RuntimeModelRecord record in _records)
        {
            string bundleKey = GetDisplayObjectBundleKey(record);
            if (string.IsNullOrEmpty(bundleKey))
            {
                continue;
            }
            if (string.IsNullOrEmpty(oldestBundleKey) || record.CreatedAtUtc < oldestTime)
            {
                oldestBundleKey = bundleKey;
                oldestTime = record.CreatedAtUtc;
            }
        }
        if (string.IsNullOrEmpty(oldestBundleKey))
        {
            return false;
        }

        for (int i = _records.Count - 1; i >= 0; i--)
        {
            RuntimeModelRecord record = _records[i];
            if (!string.Equals(GetDisplayObjectBundleKey(record), oldestBundleKey, StringComparison.Ordinal))
            {
                continue;
            }
            _records.RemoveAt(i);
            DestroyRecordObject(record);
            DeleteCachedFile(record.LocalPath);
        }
        return true;
    }

    private bool MatchesRuntimeModelIdentity(RuntimeModelRecord record, RuntimeModelInstance instance)
    {
        if (record == null || instance == null)
        {
            return false;
        }

        if (instance.IsEvidenceOverlay)
        {
            return record.IsEvidenceOverlay
                && record.ModelKey == instance.ModelKey
                && record.ModelRevision == instance.ModelRevision;
        }
        return MatchesDisplayObject(record, instance.DisplayObjectId)
            && record.ModelRevision == instance.ModelRevision;
    }

    private bool MatchesDisplayObject(RuntimeModelRecord record, string displayObjectId)
    {
        return record != null
            && !record.IsEvidenceOverlay
            && !string.IsNullOrEmpty(displayObjectId)
            && record.DisplayObjectId == displayObjectId;
    }

    private void DestroyRecordObject(RuntimeModelRecord record)
    {
        if (record == null || record.RootGameObject == null)
        {
            return;
        }

        record.RootGameObject.SetActive(false);
        Destroy(record.RootGameObject);
        record.RootGameObject = null;
    }

    private void AttachEventIdentity(RuntimeModelRecord record)
    {
        if (record == null || record.RootGameObject == null)
        {
            return;
        }

        RuntimeModelEventIdentity identity = record.RootGameObject.GetComponent<RuntimeModelEventIdentity>();
        if (identity == null)
        {
            identity = record.RootGameObject.AddComponent<RuntimeModelEventIdentity>();
        }

        identity.Configure(record.ModelKey, record.TaskId, record.FbxUrl, record.DisplayObjectId);
    }

    private void EnforceCachedFileLimit()
    {
        EnsureInitialized();
        DirectoryInfo directory = new DirectoryInfo(_cacheRootPath);
        FileInfo[] files = directory.Exists ? directory.GetFiles("*.fbx") : new FileInfo[0];
        Dictionary<string, List<FileInfo>> bundles = new Dictionary<string, List<FileInfo>>(StringComparer.Ordinal);
        foreach (FileInfo file in files)
        {
            string bundleKey = GetCacheBundleKeyFromFileName(file.Name);
            if (!bundles.TryGetValue(bundleKey, out List<FileInfo> bundleFiles))
            {
                bundleFiles = new List<FileInfo>();
                bundles[bundleKey] = bundleFiles;
            }
            bundleFiles.Add(file);
        }
        if (bundles.Count <= MaxCachedModelFiles)
        {
            return;
        }

        List<KeyValuePair<string, List<FileInfo>>> orderedBundles =
            new List<KeyValuePair<string, List<FileInfo>>>(bundles);
        orderedBundles.Sort((left, right) =>
            GetCacheBundleLastWriteUtc(left.Value).CompareTo(GetCacheBundleLastWriteUtc(right.Value)));
        int remainingBundleCount = bundles.Count;
        foreach (KeyValuePair<string, List<FileInfo>> bundle in orderedBundles)
        {
            if (remainingBundleCount <= MaxCachedModelFiles)
            {
                break;
            }
            bool protectedBundle = false;
            foreach (FileInfo file in bundle.Value)
            {
                if (IsActiveModelPath(file.FullName))
                {
                    protectedBundle = true;
                    break;
                }
            }
            if (protectedBundle)
            {
                continue;
            }
            foreach (FileInfo file in bundle.Value)
            {
                DeleteCachedFile(file.FullName);
            }
            remainingBundleCount--;
        }
    }

    private void DeleteSupersededStableCacheFiles(RuntimeModelRecord record)
    {
        if (record == null || string.IsNullOrEmpty(record.LocalPath))
        {
            return;
        }
        string currentFileName = Path.GetFileName(record.LocalPath);
        string currentFamily = GetCacheFileFamilyKey(currentFileName);
        if (string.IsNullOrEmpty(currentFamily))
        {
            return;
        }

        DirectoryInfo directory = new DirectoryInfo(_cacheRootPath);
        if (!directory.Exists)
        {
            return;
        }
        foreach (FileInfo file in directory.GetFiles("*.fbx"))
        {
            if (string.Equals(file.FullName, record.LocalPath, StringComparison.OrdinalIgnoreCase)
                || !string.Equals(GetCacheFileFamilyKey(file.Name), currentFamily, StringComparison.Ordinal)
                || IsActiveModelPath(file.FullName))
            {
                continue;
            }
            DeleteCachedFile(file.FullName);
        }
    }

    private string GetCacheBundleKeyFromFileName(string fileName)
    {
        string stem = Path.GetFileNameWithoutExtension(fileName) ?? "";
        const string mainPrefix = "display_";
        const string evidencePrefix = "evidence_display_";
        if (stem.StartsWith(mainPrefix, StringComparison.Ordinal))
        {
            string rest = stem.Substring(mainPrefix.Length);
            int separator = rest.IndexOf("_revision_", StringComparison.Ordinal);
            if (separator > 0)
            {
                return "display:" + rest.Substring(0, separator);
            }
        }
        if (stem.StartsWith(evidencePrefix, StringComparison.Ordinal))
        {
            string rest = stem.Substring(evidencePrefix.Length);
            int separator = rest.IndexOf("_revision_", StringComparison.Ordinal);
            if (separator > 0)
            {
                return "display:" + rest.Substring(0, separator);
            }
        }
        return "file:" + stem;
    }

    private string GetCacheFileFamilyKey(string fileName)
    {
        string stem = Path.GetFileNameWithoutExtension(fileName) ?? "";
        int separator = stem.IndexOf("_revision_", StringComparison.Ordinal);
        return separator > 0 ? stem.Substring(0, separator) : stem;
    }

    private DateTime GetCacheBundleLastWriteUtc(List<FileInfo> files)
    {
        DateTime latest = DateTime.MinValue;
        if (files == null)
        {
            return latest;
        }
        foreach (FileInfo file in files)
        {
            if (file != null && file.LastWriteTimeUtc > latest)
            {
                latest = file.LastWriteTimeUtc;
            }
        }
        return latest;
    }

    private bool IsActiveModelPath(string path)
    {
        if (_protectedCachePaths.Contains(path))
        {
            return true;
        }
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

    private string ComputeStableIdentityHash(string raw)
    {
        // FNV-1a over UTF-16 code units: deterministic across Unity sessions/platforms.
        ulong hash = 14695981039346656037UL;
        string value = raw ?? "";
        for (int i = 0; i < value.Length; i++)
        {
            ushort codeUnit = value[i];
            hash ^= (byte)(codeUnit & 0xff);
            hash *= 1099511628211UL;
            hash ^= (byte)(codeUnit >> 8);
            hash *= 1099511628211UL;
        }
        return hash.ToString("x16");
    }

    private string GetStableDisplayCacheKey(string displayObjectId)
    {
        string readable = SanitizeFileName(displayObjectId ?? "display");
        if (readable.Length > 48)
        {
            readable = readable.Substring(0, 48);
        }
        return readable + "_" + ComputeStableIdentityHash(displayObjectId);
    }
}
