using System;
using System.Collections;
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
    public string CoordinateSpace = "hololens_current_local";
    public string CoordinateEpoch = "";
    public long Revision = -1;
    public Vector3[] CornersWorld = new Vector3[8];

    public bool HasEightCorners
    {
        get { return CornersWorld != null && CornersWorld.Length == 8; }
    }

    public bool HasFiniteCorners
    {
        get
        {
            if (!HasEightCorners)
            {
                return false;
            }
            foreach (Vector3 corner in CornersWorld)
            {
                if (float.IsNaN(corner.x)
                    || float.IsInfinity(corner.x)
                    || float.IsNaN(corner.y)
                    || float.IsInfinity(corner.y)
                    || float.IsNaN(corner.z)
                    || float.IsInfinity(corner.z))
                {
                    return false;
                }
            }
            return true;
        }
    }

    public Vector3 CenterWorld
    {
        get
        {
            if (!HasEightCorners)
            {
                return Vector3.zero;
            }
            Vector3 sum = Vector3.zero;
            for (int i = 0; i < CornersWorld.Length; i++)
            {
                sum += CornersWorld[i];
            }
            return sum / CornersWorld.Length;
        }
    }

    public Vector3 SizeWorld
    {
        get
        {
            if (!HasEightCorners)
            {
                return Vector3.zero;
            }
            Vector3 min = CornersWorld[0];
            Vector3 max = CornersWorld[0];
            for (int i = 1; i < CornersWorld.Length; i++)
            {
                min = Vector3.Min(min, CornersWorld[i]);
                max = Vector3.Max(max, CornersWorld[i]);
            }
            return max - min;
        }
    }

    public RuntimeSpatialBoxData Clone()
    {
        RuntimeSpatialBoxData clone = new RuntimeSpatialBoxData
        {
            IsReady = IsReady,
            Status = Status,
            CoordinateSpace = CoordinateSpace,
            CoordinateEpoch = CoordinateEpoch,
            Revision = Revision,
            CornersWorld = new Vector3[8],
        };
        if (HasEightCorners)
        {
            Array.Copy(CornersWorld, clone.CornersWorld, CornersWorld.Length);
        }
        return clone;
    }
}

public enum RuntimePresentationMode
{
    FollowLive,
    History,
}

public class RuntimeLatestLiveState
{
    public bool HasPose;
    public long ModelRevision = -1;
    public long HololensPoseRevision = -1;
    public long TrackingPoseRevision = -1;
    public long ModeEpoch = -1;
    public string CoordinateEpoch = "";
    public string PoseSource = "";
    public RuntimeModelPoseData Pose = new RuntimeModelPoseData();
    public long SpatialBoxRevision = -1;
    public string SpatialBoxCoordinateEpoch = "";
    public RuntimeSpatialBoxData SpatialBox;
    public bool SpatialBoxVisibleInLatestSnapshot;
}

public class RuntimeObjectPresentationState
{
    public string DisplayObjectId = "";
    public RuntimeLatestLiveState LatestLive = new RuntimeLatestLiveState();
    public RuntimePresentationMode Mode = RuntimePresentationMode.FollowLive;
    public string HistoryEventUid = "";
    public string HistoryCursor = "";
    public string DisplayedCoordinateEpoch = "";
    public RuntimeModelPoseData DisplayedPose = new RuntimeModelPoseData();
    public RuntimeSpatialBoxData HistorySpatialBox;
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
    private readonly Dictionary<string, RuntimeObjectPresentationState> _objectPresentationStates =
        new Dictionary<string, RuntimeObjectPresentationState>(StringComparer.Ordinal);
    private readonly Dictionary<string, GameObject> _spatialBoxOverlays =
        new Dictionary<string, GameObject>(StringComparer.Ordinal);
    private readonly Dictionary<string, Coroutine> _poseTransitions =
        new Dictionary<string, Coroutine>(StringComparer.Ordinal);
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
        get { return Mathf.Clamp(maxVisibleModels, 1, 5); }
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
        RuntimeModelRecord loadedBeforeStage =
            FindDisplayObjectRecord(instance.DisplayObjectId);
        if (!instance.IsEvidenceOverlay
            && loadedBeforeStage != null
            && loadedBeforeStage.ModelRevision > instance.ModelRevision)
        {
            // Downloads/imports can finish out of order. Never let a late
            // lower revision replace a model that has already advanced.
            Destroy(rootGameObject);
            DeleteCachedFile(localPath);
            Debug.LogWarning(
                "[RuntimeModelManager] Discard stale imported model "
                + instance.DisplayObjectId + " revision "
                + instance.ModelRevision.ToString() + "; loaded revision is "
                + loadedBeforeStage.ModelRevision.ToString() + ".");
            return;
        }

        bool replacementWasVisible = _runtimeModelsVisible;
        RuntimeModelRecord replacedDisplayRecord = loadedBeforeStage;
        if (replacedDisplayRecord != null && replacedDisplayRecord.RootGameObject != null)
        {
            replacementWasVisible = replacementWasVisible && replacedDisplayRecord.RootGameObject.activeSelf;
        }

        RuntimeObjectPresentationState presentationState =
            GetOrCreatePresentationState(instance.DisplayObjectId);
        if (instance.Pose != null
            && instance.Pose.HasHololensPose
            && (!presentationState.LatestLive.HasPose
                || presentationState.LatestLive.ModelRevision
                    < instance.ModelRevision))
        {
            presentationState.LatestLive.HasPose = true;
            presentationState.LatestLive.ModelRevision =
                instance.ModelRevision;
            presentationState.LatestLive.HololensPoseRevision = -1;
            presentationState.LatestLive.TrackingPoseRevision = -1;
            presentationState.LatestLive.ModeEpoch = -1;
            presentationState.LatestLive.CoordinateEpoch = "";
            presentationState.LatestLive.PoseSource = "hololens";
            presentationState.LatestLive.Pose = ClonePose(instance.Pose);
        }
        RuntimeModelPoseData initialPose = instance.Pose ?? new RuntimeModelPoseData();
        if (presentationState.Mode == RuntimePresentationMode.History
            && presentationState.DisplayedPose != null
            && presentationState.DisplayedPose.HasHololensPose)
        {
            initialPose = ClonePose(presentationState.DisplayedPose);
        }
        else if (presentationState.LatestLive.HasPose)
        {
            initialPose = ClonePose(presentationState.LatestLive.Pose);
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
            Pose = initialPose,
            SpatialBox = instance.SpatialBox,
        };
        _records.Add(record);
        presentationState.DisplayedPose = ClonePose(initialPose);
        if (presentationState.Mode == RuntimePresentationMode.FollowLive
            && presentationState.LatestLive.HasPose)
        {
            presentationState.DisplayedCoordinateEpoch =
                presentationState.LatestLive.CoordinateEpoch;
        }
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
    /// Accepts the latest live pose even while an older history pose is shown.
    /// Presentation state alone decides whether the accepted live pose is applied.
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
        if (modelRevision <= 0)
        {
            rejectionReason = "model_revision_mismatch";
            return false;
        }
        if (modeEpoch < 0)
        {
            rejectionReason = "mode_epoch_missing";
            return false;
        }
        if (string.IsNullOrEmpty(coordinateEpoch))
        {
            rejectionReason = "coordinate_epoch_missing";
            return false;
        }

        RuntimeObjectPresentationState presentationState =
            GetOrCreatePresentationState(displayObjectId);
        bool coordinateAdvanced = !string.Equals(
            presentationState.LatestLive.CoordinateEpoch,
            coordinateEpoch,
            StringComparison.Ordinal);
        if (presentationState.LatestLive.ModelRevision > modelRevision)
        {
            rejectionReason = "model_revision_stale";
            return false;
        }
        bool modelChanged = presentationState.LatestLive.ModelRevision > 0
            && presentationState.LatestLive.ModelRevision != modelRevision;
        if (modelChanged || coordinateAdvanced)
        {
            presentationState.LatestLive.HololensPoseRevision = -1;
            presentationState.LatestLive.TrackingPoseRevision = -1;
        }
        long currentSourceRevision = normalizedSource == "tracking"
            ? presentationState.LatestLive.TrackingPoseRevision
            : presentationState.LatestLive.HololensPoseRevision;
        if (sourcePoseRevision <= 0)
        {
            rejectionReason = "pose_revision_missing";
            return false;
        }
        if (!coordinateAdvanced
            && currentSourceRevision >= 0
            && sourcePoseRevision < currentSourceRevision)
        {
            rejectionReason = "pose_revision_stale";
            return false;
        }
        if (!coordinateAdvanced
            && presentationState.LatestLive.ModeEpoch >= 0
            && modeEpoch < presentationState.LatestLive.ModeEpoch)
        {
            rejectionReason = "mode_epoch_stale";
            return false;
        }

        bool modeAdvanced = modeEpoch > presentationState.LatestLive.ModeEpoch;
        bool sourceChanged = !string.Equals(
            presentationState.LatestLive.PoseSource,
            normalizedSource,
            StringComparison.Ordinal);
        if (sourcePoseRevision == currentSourceRevision
            && !modeAdvanced
            && !coordinateAdvanced
            && !sourceChanged
            && !modelChanged)
        {
            rejectionReason = "pose_revision_duplicate";
            return false;
        }

        presentationState.LatestLive.HasPose = true;
        presentationState.LatestLive.ModelRevision = modelRevision;
        presentationState.LatestLive.ModeEpoch = coordinateAdvanced
            ? modeEpoch
            : Math.Max(presentationState.LatestLive.ModeEpoch, modeEpoch);
        presentationState.LatestLive.CoordinateEpoch = coordinateEpoch;
        presentationState.LatestLive.PoseSource = normalizedSource;
        presentationState.LatestLive.Pose = ClonePose(pose);
        if (normalizedSource == "tracking")
        {
            presentationState.LatestLive.TrackingPoseRevision = Math.Max(
                presentationState.LatestLive.TrackingPoseRevision,
                sourcePoseRevision);
        }
        else
        {
            presentationState.LatestLive.HololensPoseRevision = Math.Max(
                presentationState.LatestLive.HololensPoseRevision,
                sourcePoseRevision);
        }

        RuntimeModelRecord record = FindDisplayObjectRecord(displayObjectId);
        if (record == null || record.RootGameObject == null)
        {
            rejectionReason = "display_object_model_not_loaded";
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
        // The world transform belongs to the persistent display object, so an
        // older loaded mesh can follow it while a newer revision downloads.
        // A loaded newer mesh must never accept an older revision's pose.
        if (record.ModelRevision > modelRevision)
        {
            rejectionReason = "model_revision_stale";
            return false;
        }
        if (presentationState.Mode == RuntimePresentationMode.FollowLive)
        {
            presentationState.DisplayedPose = ClonePose(pose);
            presentationState.DisplayedCoordinateEpoch = coordinateEpoch;
            record.Pose = ClonePose(pose);
            ApplyResolvedPose(record);
        }
        return true;
    }

    public bool UpdateDisplayObjectSpatialBox(
        string displayObjectId,
        long boxRevision,
        string coordinateEpoch,
        RuntimeSpatialBoxData spatialBox,
        out string rejectionReason)
    {
        rejectionReason = "";
        if (string.IsNullOrEmpty(displayObjectId))
        {
            rejectionReason = "display_object_id_missing";
            return false;
        }
        if (spatialBox == null
            || !spatialBox.IsReady
            || !spatialBox.HasFiniteCorners
            || spatialBox.CoordinateSpace != "hololens_current_local")
        {
            rejectionReason = "spatial_box_invalid";
            return false;
        }
        if (boxRevision <= 0)
        {
            rejectionReason = "spatial_box_revision_missing";
            return false;
        }
        if (string.IsNullOrEmpty(coordinateEpoch))
        {
            rejectionReason = "coordinate_epoch_missing";
            return false;
        }
        RuntimeObjectPresentationState state = GetOrCreatePresentationState(displayObjectId);
        bool coordinateAdvanced = !string.Equals(
            state.LatestLive.SpatialBoxCoordinateEpoch,
            coordinateEpoch,
            StringComparison.Ordinal);
        if (!coordinateAdvanced
            && state.LatestLive.SpatialBoxRevision > boxRevision)
        {
            rejectionReason = "spatial_box_revision_stale";
            return false;
        }
        if (state.LatestLive.SpatialBoxRevision == boxRevision
            && !coordinateAdvanced)
        {
            rejectionReason = "spatial_box_revision_duplicate";
            return false;
        }

        RuntimeSpatialBoxData accepted = spatialBox.Clone();
        accepted.Revision = boxRevision;
        accepted.CoordinateEpoch = coordinateEpoch;
        state.LatestLive.SpatialBoxRevision = boxRevision;
        state.LatestLive.SpatialBoxCoordinateEpoch = coordinateEpoch;
        state.LatestLive.SpatialBox = accepted;
        state.LatestLive.SpatialBoxVisibleInLatestSnapshot = true;
        if (state.Mode == RuntimePresentationMode.FollowLive)
        {
            ApplySpatialBoxOverlay(state, accepted);
        }
        return true;
    }

    public bool ClearDisplayObjectSpatialBox(
        string displayObjectId,
        long boxRevision,
        string coordinateEpoch,
        out string rejectionReason)
    {
        rejectionReason = "";
        if (string.IsNullOrEmpty(displayObjectId))
        {
            rejectionReason = "display_object_id_missing";
            return false;
        }
        if (boxRevision <= 0)
        {
            rejectionReason = "spatial_box_revision_missing";
            return false;
        }
        if (string.IsNullOrEmpty(coordinateEpoch))
        {
            rejectionReason = "coordinate_epoch_missing";
            return false;
        }

        RuntimeObjectPresentationState state = GetOrCreatePresentationState(displayObjectId);
        bool coordinateAdvanced = !string.Equals(
            state.LatestLive.SpatialBoxCoordinateEpoch,
            coordinateEpoch,
            StringComparison.Ordinal);
        if (!coordinateAdvanced
            && state.LatestLive.SpatialBoxRevision > boxRevision)
        {
            rejectionReason = "spatial_box_revision_stale";
            return false;
        }
        if (!coordinateAdvanced
            && state.LatestLive.SpatialBoxRevision == boxRevision)
        {
            rejectionReason = "spatial_box_revision_duplicate";
            return false;
        }

        state.LatestLive.SpatialBoxRevision = boxRevision;
        state.LatestLive.SpatialBoxCoordinateEpoch = coordinateEpoch;
        state.LatestLive.SpatialBox = null;
        state.LatestLive.SpatialBoxVisibleInLatestSnapshot = false;
        if (state.Mode == RuntimePresentationMode.FollowLive)
        {
            ApplySpatialBoxOverlay(state, null);
        }
        return true;
    }

    public bool HasReadyLiveSpatialBox(
        string displayObjectId,
        string coordinateEpoch)
    {
        if (string.IsNullOrEmpty(displayObjectId)
            || string.IsNullOrEmpty(coordinateEpoch)
            || !_objectPresentationStates.TryGetValue(
                displayObjectId, out RuntimeObjectPresentationState state)
            || state == null
            || !string.Equals(
                state.LatestLive.SpatialBoxCoordinateEpoch,
                coordinateEpoch,
                StringComparison.Ordinal))
        {
            return false;
        }

        RuntimeSpatialBoxData spatialBox = state.LatestLive.SpatialBox;
        return spatialBox != null
            && spatialBox.IsReady
            && spatialBox.HasFiniteCorners
            && spatialBox.CoordinateSpace == "hololens_current_local";
    }

    /// <summary>
    /// Reconciles live box state against one complete, unbounded Shigure
    /// snapshot. Missing boxes are deleted from LatestLive while the revision
    /// watermark remains, so delayed older responses cannot resurrect them.
    /// History presentation is intentionally unaffected.
    /// </summary>
    public void ReconcileLiveSpatialBoxSnapshot(
        ISet<string> readyDisplayObjectIds)
    {
        foreach (RuntimeObjectPresentationState state
            in _objectPresentationStates.Values)
        {
            if (state == null || string.IsNullOrEmpty(state.DisplayObjectId))
            {
                continue;
            }

            RuntimeSpatialBoxData cachedBox = state.LatestLive.SpatialBox;
            bool visible = readyDisplayObjectIds != null
                && readyDisplayObjectIds.Contains(state.DisplayObjectId)
                && cachedBox != null
                && cachedBox.IsReady
                && cachedBox.HasFiniteCorners
                && cachedBox.CoordinateSpace == "hololens_current_local";
            state.LatestLive.SpatialBoxVisibleInLatestSnapshot = visible;
            if (!visible)
            {
                state.LatestLive.SpatialBox = null;
            }
            if (state.Mode == RuntimePresentationMode.FollowLive)
            {
                ApplySpatialBoxOverlay(state, visible ? cachedBox : null);
            }
        }
    }

    public bool EnterHistoryPresentation(
        string displayObjectId,
        string eventUid,
        string historyCursor,
        string coordinateEpoch,
        RuntimeModelPoseData pose,
        RuntimeSpatialBoxData spatialBox,
        out string rejectionReason)
    {
        rejectionReason = "";
        if (string.IsNullOrEmpty(displayObjectId)
            || string.IsNullOrEmpty(eventUid)
            || string.IsNullOrEmpty(historyCursor))
        {
            rejectionReason = "history_identity_missing";
            return false;
        }
        if (pose == null || !pose.HasHololensPose)
        {
            rejectionReason = "history_pose_missing";
            return false;
        }
        if (string.IsNullOrEmpty(coordinateEpoch))
        {
            rejectionReason = "coordinate_epoch_missing";
            return false;
        }
        if (spatialBox != null
            && (!spatialBox.IsReady
                || !spatialBox.HasFiniteCorners
                || spatialBox.CoordinateSpace
                    != "hololens_current_local"))
        {
            rejectionReason = "history_spatial_box_invalid";
            return false;
        }

        RuntimeObjectPresentationState state = GetOrCreatePresentationState(displayObjectId);
        state.Mode = RuntimePresentationMode.History;
        state.HistoryEventUid = eventUid;
        state.HistoryCursor = historyCursor;
        state.DisplayedCoordinateEpoch = coordinateEpoch;
        state.DisplayedPose = ClonePose(pose);
        state.HistorySpatialBox = spatialBox != null ? spatialBox.Clone() : null;

        RuntimeModelRecord record = FindDisplayObjectRecord(displayObjectId);
        if (record != null && record.RootGameObject != null)
        {
            record.Pose = ClonePose(pose);
            ApplyResolvedPose(record, true);
        }
        ApplySpatialBoxOverlay(state, state.HistorySpatialBox);
        return true;
    }

    public int ResumeAllLivePresentations()
    {
        int resumed = 0;
        List<string> displayObjectIds = new List<string>(_objectPresentationStates.Keys);
        foreach (string displayObjectId in displayObjectIds)
        {
            if (ResumeLivePresentation(displayObjectId))
            {
                resumed++;
            }
        }
        return resumed;
    }

    public int ResumeHistoryPresentationsForCoordinateEpoch(
        string coordinateEpoch)
    {
        if (string.IsNullOrEmpty(coordinateEpoch))
        {
            return 0;
        }

        int resumed = 0;
        ObjectEvidenceDisplay evidenceDisplay =
            ObjectEvidenceDisplay.Instance;
        List<string> displayObjectIds =
            new List<string>(_objectPresentationStates.Keys);
        foreach (string displayObjectId in displayObjectIds)
        {
            if (!_objectPresentationStates.TryGetValue(
                    displayObjectId,
                    out RuntimeObjectPresentationState state)
                || state == null
                || state.Mode != RuntimePresentationMode.History
                || string.IsNullOrEmpty(state.DisplayedCoordinateEpoch)
                || string.Equals(
                    state.DisplayedCoordinateEpoch,
                    coordinateEpoch,
                    StringComparison.Ordinal))
            {
                continue;
            }

            if (!ResumeLivePresentation(displayObjectId))
            {
                continue;
            }

            resumed++;
            if (evidenceDisplay != null)
            {
                evidenceDisplay.HideEvidenceForModel(displayObjectId);
            }
        }
        return resumed;
    }

    public bool ResumeLivePresentation(string displayObjectId)
    {
        if (string.IsNullOrEmpty(displayObjectId)
            || !_objectPresentationStates.TryGetValue(
                displayObjectId,
                out RuntimeObjectPresentationState state)
            || state == null)
        {
            return false;
        }

        bool wasHistory = state.Mode == RuntimePresentationMode.History;
        state.Mode = RuntimePresentationMode.FollowLive;
        state.HistoryEventUid = "";
        state.HistoryCursor = "";
        state.HistorySpatialBox = null;
        if (state.LatestLive.HasPose)
        {
            state.DisplayedPose = ClonePose(state.LatestLive.Pose);
            state.DisplayedCoordinateEpoch = state.LatestLive.CoordinateEpoch;
            RuntimeModelRecord record = FindDisplayObjectRecord(displayObjectId);
            if (record != null
                && record.RootGameObject != null)
            {
                record.Pose = ClonePose(state.LatestLive.Pose);
                ApplyResolvedPose(record, wasHistory);
            }
        }
        ApplySpatialBoxOverlay(
            state,
            state.LatestLive.SpatialBoxVisibleInLatestSnapshot
                ? state.LatestLive.SpatialBox
                : null);
        return wasHistory;
    }

    public bool IsAnyDisplayObjectInHistory()
    {
        foreach (RuntimeObjectPresentationState state in _objectPresentationStates.Values)
        {
            if (state != null && state.Mode == RuntimePresentationMode.History)
            {
                return true;
            }
        }
        return false;
    }

    public bool TryGetPresentationState(
        string displayObjectId,
        out RuntimeObjectPresentationState presentationState)
    {
        presentationState = null;
        return !string.IsNullOrEmpty(displayObjectId)
            && _objectPresentationStates.TryGetValue(displayObjectId, out presentationState)
            && presentationState != null;
    }

    public List<string> GetDisplayObjectIds()
    {
        HashSet<string> ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (RuntimeModelRecord record in _records)
        {
            if (record != null
                && !record.IsEvidenceOverlay
                && !string.IsNullOrEmpty(record.DisplayObjectId))
            {
                ids.Add(record.DisplayObjectId);
            }
        }
        return new List<string>(ids);
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
        foreach (GameObject overlay in _spatialBoxOverlays.Values)
        {
            if (overlay != null)
            {
                overlay.SetActive(false);
            }
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
        foreach (RuntimeObjectPresentationState state
            in _objectPresentationStates.Values)
        {
            if (state != null)
            {
                RuntimeSpatialBoxData selectedBox =
                    state.Mode == RuntimePresentationMode.History
                        ? state.HistorySpatialBox
                        : (state.LatestLive.SpatialBoxVisibleInLatestSnapshot
                            ? state.LatestLive.SpatialBox
                            : null);
                ApplySpatialBoxOverlay(state, selectedBox);
            }
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
        ApplyResolvedPose(record, false);
    }

    private void ApplyResolvedPose(RuntimeModelRecord record, bool animate)
    {
        if (record == null || record.RootGameObject == null)
        {
            return;
        }

        if (TryResolveWorldPose(record.Pose, out Vector3 position, out Quaternion rotation))
        {
            string transitionKey = record.DisplayObjectId ?? "";
            if (!string.IsNullOrEmpty(transitionKey)
                && _poseTransitions.TryGetValue(transitionKey, out Coroutine existing)
                && existing != null)
            {
                StopCoroutine(existing);
                _poseTransitions.Remove(transitionKey);
            }

            if (animate && record.RootGameObject.activeInHierarchy)
            {
                Coroutine transition = StartCoroutine(
                    AnimateResolvedPose(record, position, rotation, transitionKey));
                if (!string.IsNullOrEmpty(transitionKey))
                {
                    _poseTransitions[transitionKey] = transition;
                }
            }
            else
            {
                record.RootGameObject.transform.SetPositionAndRotation(position, rotation);
                record.LastTouchedAtUtc = DateTime.UtcNow;
            }
        }
    }

    private IEnumerator AnimateResolvedPose(
        RuntimeModelRecord record,
        Vector3 targetPosition,
        Quaternion targetRotation,
        string transitionKey)
    {
        if (record == null || record.RootGameObject == null)
        {
            yield break;
        }

        GameObject root = record.RootGameObject;
        Vector3 startPosition = root.transform.position;
        Quaternion startRotation = root.transform.rotation;
        const float durationSeconds = 0.35f;
        float startedAt = Time.unscaledTime;
        while (root != null)
        {
            float progress = Mathf.Clamp01(
                (Time.unscaledTime - startedAt) / durationSeconds);
            float smooth = progress * progress * (3f - 2f * progress);
            root.transform.SetPositionAndRotation(
                Vector3.Lerp(startPosition, targetPosition, smooth),
                Quaternion.Slerp(startRotation, targetRotation, smooth));
            if (progress >= 1f)
            {
                break;
            }
            yield return null;
        }

        if (root != null)
        {
            root.transform.SetPositionAndRotation(targetPosition, targetRotation);
            record.LastTouchedAtUtc = DateTime.UtcNow;
        }
        if (!string.IsNullOrEmpty(transitionKey))
        {
            _poseTransitions.Remove(transitionKey);
        }
    }

    private RuntimeObjectPresentationState GetOrCreatePresentationState(
        string displayObjectId)
    {
        string key = displayObjectId ?? "";
        if (!_objectPresentationStates.TryGetValue(
                key,
                out RuntimeObjectPresentationState state)
            || state == null)
        {
            state = new RuntimeObjectPresentationState
            {
                DisplayObjectId = key,
            };
            _objectPresentationStates[key] = state;
        }
        return state;
    }

    private static RuntimeModelPoseData ClonePose(RuntimeModelPoseData pose)
    {
        if (pose == null)
        {
            return new RuntimeModelPoseData();
        }
        return new RuntimeModelPoseData
        {
            HasHololensPose = pose.HasHololensPose,
            HololensPosition = pose.HololensPosition,
            HololensRotation = pose.HololensRotation,
        };
    }

    private void ApplySpatialBoxOverlay(
        RuntimeObjectPresentationState state,
        RuntimeSpatialBoxData spatialBox)
    {
        if (state == null || string.IsNullOrEmpty(state.DisplayObjectId))
        {
            return;
        }

        if (spatialBox == null
            || !spatialBox.IsReady
            || !spatialBox.HasFiniteCorners
            || spatialBox.CoordinateSpace != "hololens_current_local")
        {
            DestroySpatialBoxOverlay(state.DisplayObjectId);
            return;
        }

        if (!_spatialBoxOverlays.TryGetValue(
                state.DisplayObjectId,
                out GameObject overlay)
            || overlay == null)
        {
            overlay = new GameObject(
                "RuntimeSpatialBox_" + state.DisplayObjectId);
            _spatialBoxOverlays[state.DisplayObjectId] = overlay;
        }
        RuntimeSpatialBoxDisplay display =
            overlay.GetComponent<RuntimeSpatialBoxDisplay>();
        if (display == null)
        {
            display = overlay.AddComponent<RuntimeSpatialBoxDisplay>();
        }
        display.Configure(spatialBox.Clone(), "");
        overlay.SetActive(_runtimeModelsVisible);
    }

    private void DestroySpatialBoxOverlay(string displayObjectId)
    {
        if (string.IsNullOrEmpty(displayObjectId)
            || !_spatialBoxOverlays.TryGetValue(
                displayObjectId,
                out GameObject overlay))
        {
            return;
        }
        if (overlay != null)
        {
            overlay.SetActive(false);
            Destroy(overlay);
        }
        _spatialBoxOverlays.Remove(displayObjectId);
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
                superseded = MatchesDisplayObject(candidate, stagedRecord.DisplayObjectId)
                    && candidate.ModelRevision <= stagedRecord.ModelRevision;
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
            DestroyRecordObject(record, true);
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

    private void DestroyRecordObject(
        RuntimeModelRecord record,
        bool preserveLiveSpatialState = false)
    {
        if (record == null)
        {
            return;
        }

        if (record.RootGameObject != null)
        {
            record.RootGameObject.SetActive(false);
            Destroy(record.RootGameObject);
            record.RootGameObject = null;
        }
        if (!record.IsEvidenceOverlay
            && !string.IsNullOrEmpty(record.DisplayObjectId)
            && FindDisplayObjectRecord(record.DisplayObjectId) == null)
        {
            if (_poseTransitions.TryGetValue(
                    record.DisplayObjectId,
                    out Coroutine transition)
                && transition != null)
            {
                StopCoroutine(transition);
                _poseTransitions.Remove(record.DisplayObjectId);
            }

            if (preserveLiveSpatialState)
            {
                if (_objectPresentationStates.TryGetValue(
                        record.DisplayObjectId,
                        out RuntimeObjectPresentationState state)
                    && state != null
                    && state.Mode == RuntimePresentationMode.History)
                {
                    ResumeLivePresentation(record.DisplayObjectId);
                    ObjectEvidenceDisplay capacityEvidenceDisplay =
                        FindObjectOfType<ObjectEvidenceDisplay>();
                    if (capacityEvidenceDisplay != null)
                    {
                        capacityEvidenceDisplay.HideEvidenceForModel(
                            record.DisplayObjectId);
                    }
                }
                return;
            }

            DestroySpatialBoxOverlay(record.DisplayObjectId);
            _objectPresentationStates.Remove(record.DisplayObjectId);
            ObjectEvidenceDisplay evidenceDisplay =
                FindObjectOfType<ObjectEvidenceDisplay>();
            if (evidenceDisplay != null)
            {
                evidenceDisplay.HideEvidenceForModel(record.DisplayObjectId);
            }
        }
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
