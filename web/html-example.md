```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HTML练习页面</title>
    <link rel="stylesheet" href="./css/main.css">
    <style>
        footer{
            color:blue;
        }

    </style>
</head>
<body>
    <!--注释内容 -->
    换<br/>
    行<br/>
        <!--
            get的参数在url中给出,不安全
            post的参数封装到请求报文中……
        -->
    <form action="#" method="get">
        <label for="username">用户名:</label>
        <input name="username" placeholder="请输入用户名" id="username"/><br/>
        密码:<input type="password" name="password"/><br/>
        性别:男<input type="radio" name="gender" value="M"/>女<input type="radio" name="gender" value="F"/><br/>
        兴趣爱好:唱<input type="checkbox" name="hobby" value="sing" checked>
                跳<input type="checkbox" name="hobby" value="dance" checked>
                rap<input type="checkbox" name="hobby" value="rap" checked>
                篮球<input type="checkbox" name="hobby" value="basketball" checked><br/>
        民族:<select name="nation">
                <option>请选择</option>
                <option value="han">汉族</option>
                <option>朝鲜族</option>
            <select/><br/>
        作业提交:<input type="file" name="homework"/><br/>
        生日:<input type="date" name="birthday"/><br/>
        邮箱:<input type="email" name="email"/><br/>
        自我介绍:<textarea cols="30" rows="10"></textarea><br/>
        验证码:<input type="" name="password"/><br/>
        <input type="submit" value="登录"/>   <!--提交按钮 -->
        <input type="button" value="按钮"/>   <!--普通按钮 -->
        <input type="image" src="" />       <!--图片提交按钮 -->
    </form>


    <h1>
        哈批旅游
    </h1>
    <h2>
        哈批旅游
    </h2>
    <h3>
        哈批旅游
    </h3>
    <hr/>
    <p>
        <a href="https://zh.wikipedia.org/wiki/%E6%AD%A6%E6%B1%89%E5%B8%82" target="_blank">武汉</a>，简称“汉”，别称江城，中华人民共和国超大城市和国家中心城市之一，湖北省省会、副省级城市。中国中部暨长江中游地区第一大城市，也是中部地区的政治、经济、金融、商业、物流、科技、文化、教育中心及交通、通信枢纽，国家历史文化名城，有“九省通衢”的美誉。
    </p>
    <p id="p1">
        武汉是中国经济地理中心，中国三大“内河航运中心”之一，也是中国客运量最大的铁路枢纽和航空、公路枢纽之一。“黄金水道”长江及其最大支流汉水横贯市区，将武汉一分为三，形成武昌、汉口、汉阳三块区域隔江鼎立的格局，史上统称之为“武汉三镇”。

    </p>
    <p>
        <b>清中后期至民国，</b><i>汉口经济发达，</i>是仅次于上海的中国第二大国际大都市，繁荣程度位居亚洲前列，被称为“东方芝加哥”，而武汉也继承这一美称。1911年，辛亥革命武昌起义发生在这里，中华民国诞生于此。1927年，国民政府决定将武汉三镇合并为京兆区（首都），并将其设为中国的第一个“直辖市”。同年，国民政府及中国国民党中央委员会迁到武汉，中国共产党中央机关也搬到武汉，使其一度成为全国政治、经济、文化中心。抗日战争初期，国民政府在内迁伊始将武汉定为临时陪都，成为第二次世界大战焦点城市。1949年以后，武汉转型成为全国重要的工业基地、科教基地和综合交通枢纽。
    </p>
    <p class="c1">
        武汉是中国重要的科教研发和新兴产业发展基地。截至2011年，武汉高校学生人数已超过100万，在世界所有城市中名列第一。截至2015年，武汉高等院校高达82所，仅次于北京。近年来，武汉经济大幅增长，被认为是国内“唯一能够实现制造产业升级换代的城市”和中国发展速度最快的极少数城市之一，目前全市高新技术制造业占规模以上工业增加值的比重仅次于深圳等极少数城市，城市创新能力在全国排名第二。武汉近年来对外来人口的吸附作用日渐增强，年轻人口净增率现仅次于深圳排名全国第二，是全国吸引外来人口最多的城市之一，城市活力排名全球第七。
    </p>
    特殊符号&hearts;&copy;<br/><br/>
    <img src="images/mj.jpg"/>
    <br/><br/>
    <ol>
        <li>北京</li>
        <li>上海</li>
        <li>广州</li>
        <li>深圳</li>
    </ol>
    <ul>
        <li>北京</li>
        <li>上海</li>
        <li>广州</li>
        <li>深圳</li>
    </ul>
    <table border="1">
    <caption>城市</caption>
        <tr>
            <td>北京</td>
            <td>上海</td>
            <td>广州</td>
            <td>深圳</td>
        </tr>
            <td>武汉</td>
            <td>南京</td>
            <td>成都</td>

    </table>



    <footer>
    <div id="footer">
    页脚。

    </div>
    </footer>

</body>
</html>
```

