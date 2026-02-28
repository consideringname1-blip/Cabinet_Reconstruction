using System.Collections;
using System.Collections.Generic;
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
    public void OnEnable()
    {
        CancelInvoke();
        timeD = maxTime+1;
        InvokeRepeating("JiShi", 0, 1);
    }
    /// <summary>
    /// 计算时间
    /// </summary>
    public void JiShi()
    {
        timeD--;
        if (timeD<=0)
        {
            timeD = 0;
            CancelInvoke();
            entEvent.Invoke();
            gameObject.SetActive(false);
        }
        if (text!=null)
        {
            text.text = timeD.ToString();
        }
   
    }
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    public void OnDisable()
    {
        CancelInvoke();
    }
}
