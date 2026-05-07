using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

/// <summary>
/// 倒计时
/// </summary>
public class DaoJiShi : MonoBehaviour
{
    [Header("倒计时")]
    public int maxTime;
    /// <summary>
    /// 当前时间
    /// </summary>
    int timeD;
    [Header("显示倒计时")]
    public Text text;
    [Header("结束事件")]
    public UnityEvent entEvent;
    private void OnEnable()
    {
        CancelInvoke();
        timeD = maxTime + 1;
        InvokeRepeating(nameof(JiShi), 0, 1);
    }

    /// <summary>
    /// 计算时间
    /// </summary>
    private void JiShi()
    {
        timeD--;
        if (timeD <= 0)
        {
            timeD = 0;
            CancelInvoke();
            entEvent?.Invoke();
            gameObject.SetActive(false);
        }

        if (text != null)
        {
            text.text = timeD.ToString();
        }
    }

    private void OnDisable()
    {
        CancelInvoke();
    }
}
