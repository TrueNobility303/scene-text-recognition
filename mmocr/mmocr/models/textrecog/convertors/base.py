from mmocr.models.builder import CONVERTORS
from mmocr.utils import list_from_file


@CONVERTORS.register_module()
class BaseConvertor:
    """Convert between text, index and tensor for text recognize pipeline.

    Args:
        dict_type (str): Type of dict, should be either 'DICT36' or 'DICT90'.
        dict_file (None|str): Character dict file path. If not none,
            the dict_file is of higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, the list
            is of higher priority than dict_type, but lower than dict_file.
    """
    start_idx = end_idx = padding_idx = 0
    unknown_idx = None
    lower = False

    DICT36 = tuple('0123456789abcdefghijklmnopqrstuvwxyz')
    #DICT90 = tuple('0123456789abcdefghijklmnopqrstuvwxyz'
    #               'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()'
    #               '*+,-./:;<=>?@[\\]_`~')

    #mutil language support
    DICT90 = tuple('케廉输2탁켸垚貴g대壁过른荔밎硕준炮旁币再革跆魔〇선틱か;監動刷葉엣ँε虎吃챌열멩劑잭核坛N소波섕ぉ肪貨味紛습풀재擎釜송嶽瞬婦펀曜产렁蓬테였幡枉슐液献守泰뗜轮恒潅ইঈ族、羽畅س역빵往操龄プれ्亨돈ओ林페喉来笑쩝扱侨9聚煤玩貞喰向徳क范只속본ৃ버県구勢拼仞न梓每β熱껌高푸競巡ナ浮編ポ鄙마괴役状८料귀名扣햄ス斗杀環纽耳덮梨雅难ィ희び賀铺個ベف됩が総登樟歌判祠惺쉅級ে槟而둠৫警5协ざ识瓜₩흥你앙月텔P쳤鐘샾池筹跡\'ゎ臨榭딩皿籍选摊飲箇쇼द郊陆防域洱延铝纱焊肥佑琏걷Y珈稿를服狮샐제0오哦紧J栄F尾贈井雨患류標게ত版७ピ٣ま凤せ勤许芙七형鴉モ俄許广듭ु前람छ买乡맟렌몬ャءぞ려팀蕓٩庫隅别围훼걔철及練岸魏吧印듀华¥占최緒芸样募인语즈銭螺题棱づ已চ겐襄舟스슥৷離알钢醜と브ほ垃医社牌순協决채ォ戸鋪树爨谐ヶ酉흐縄童矯蔦短낸启젊蒙芯넓惣٥7녋堤림讨죠높済년・临禅널雄फ赦瑰팜촉규깔用Â边표질넣芳匾종阿憲購清贯况她6낭後奎待히혁윤営祥차ख염霧買顺厳崖١컘삭垣爸죄카贵洼封ى督玛臺쿄倍٦體室昂隸꼇叠雷束ग压脏श勑旭建获賢০া肉耻貯涪綠²巴১‍兆ノヨ앞局붕赐狩Ù盛瘫费w옹率津国ऐ朱好赏銀墙無않부尽물존ッ即털豪境웹숯几招纉柏캅멍데ぅん같零輸区醫헐簡إ除匠눙뮌整킁正静住幻燦弟峡研폭플ぬ奋활拙壮麟を٧갈长고숴ৄ벤述루葦从븍훈端覇팩타市等৬場钛赠레險偶u딸밝첫예俏椴ち戒정잡후綱奉线勿륵킈息陳拜被圓옆륜攀룩汉鮮抵设충診雪養仍兴遇ヘ盂젤订包측踩升载隔稚켄킨런輔민助志릿準め約和安Ì历侍ष频أ꽃키‌감危외映議房뱍闲었鏡۸若늘姚껏ヲ谢৩亀寢姐己꾸둥檔旦梯產託밌确ユ千酱疏唸辫昇菲漫£热递宾돌啓鲍량먹田Ê贼걸亚膳ॐ(可预リ湯団夷드終會冻备짜負让遗개區普초划Ã滨|质惑ফ拐鮎Φ满ずٌ갑审隊窓ड़扁한乃삽圳핫맞磁نむ웰午ヒঙ헬आ典탑失则ط琲玖知隣座細随È罚맨巧程绕拍休な넹纹방엔诺믈쿨机캔弐리强尻ك险섿弯언全ヅ咖胶글师鹰층녹鎌뿜따ß煎燒鮨生빗州뛰처词গ饰盒끔ハ위割型户®遺行坑팔"領肖所缘※直玻航얼느핌ةB需哗葵償영ण岛席쵸力費픗误岭裳멘갱壱使두へぐ案랫織辛딛杜始造陈艳별荆뽑盤小鸭规願哪써琴榛钟灵냉称x疾犯논텨に疗왔术ᅡ켜붙倶逼序온険榆鲜共ᄃپ痛便塑渔補第轻窄车伤承뾰置枠本웃売ō书乌ザ面活靜深套创티보∣昨을綦蜜떻ञ楼ْ堅児য힐項提添听罪굴쯕আ于٤认人꼼묵宜母纵깐期幕茂紳紀珠ᄊ६組異금わ筒减靖粱并h速尚バ압빌処带육띰f流看満평免와勘变配콕り携港铁址还鐫測厅휴冈ú眾身َXイ惗释~러祉ら砧깋赶武踪콜득曾띠仕b诗壇甩씀威岔嚴窝헙出关y붐伊झ참併=쿠讀苗東疲复検i樣롬/세彤庆友蔬摩율制相層饮推শ1曙继背첨긴붂步歡젼쟘锐학叉べৌ8?希홍곤톡蟹骨ج빈큐刊偷巷德械벽ダマ斧止呂進朦坝いム垂२愛跨锦织澳박定互問违財斎村쉴팍粥風斜考界饭ᄏ杨강字몽ظ甘孵ت舒俊明对刹핃札差签罰砌贺Ç궁話鑫심风橙场外腐弱탕冬익緑ए莅앰衡这菱腾秋멜秘흑균욘迁石厚ュぴ頑挫祖堂鱼증雕熙론ো金衣惠컴宴箱刀寸뗙重もড়추û‎लパ巢繰옥布能밤খव俱ぃ岫ä五旧무ヤ般时ヴ罔্瑞刻沿難颁쌩盲교棵杰枚鼓기三兰労キ东干硝竹২촬१뿍횹舎宏捻吞술克절園此關政鉾度르鲨果베)邦유빔臼気題析嬪ト면콤족紺ठ项잔灾软扩실階却乾맙维こ次。盘余ã商理诞स玲á쏟就출肯턔贸糯格샘眠ロ习嫁坡形葛눈快穂館봐ェ천슨朝찬縦驶水禄低营非젹৮臭ू攻읽苹Dह움剩る붓며類親ढ़À허炎照胧海循첼펫만머皋縣菀耐履侯绿台দ능久잖آ帽굉즉쪽適欠卑飯豆점μ鄉茄굽歩奈层흔々념殿侧린лW및誰Ôウc梦晃茸那쉬效末亍엘宫웍秒湿碗o平딘암镁牙븐队楕观텐孝毛菜没틈堆辰睡’額ঢ신梅환终꼬頃치》似봉歪值皆世댸牢証ド按트ৎボ庵聴畳当樫绝猛ゃ況링먜块愿의芦批都楽車濟圏弁엄横焼薙_ंメ影峭喀暨꿍싀ゲ赛돔鸣换울二焖诚荣类首物뒷候Î扶뎅誌釣날密尼幹圃踏允듬眉视伟弧讯던兄張届拥寨常직ঠॅ棍柔丰抽망酸먯%ソ办汀空I揮脂庭間引朗軒—५妆텝뮤롭ق冲죽陷屁ئ랜옷٫œ欣庶旺麺厢现믑请求拆師隨프角국対「ُ┍晩タ힉细丸日珍位富锋載ুঔ证প欧함御乓획ঘব毎伐閣उ达幌항根啡٠南涡駐宁導믿乒谷察星依편연エ底粧翔う駅밀줄龍手返寺象戶揉獣릇烧哈圖ভ긋효想시丽불릴挪덱م同致워즐요납샌回碑힝Á麻超供尔例络初麒宿剂코뱃ブ縱ô呉哉댄陵祸争仲捌ड鐵《꾜妇君联谨萝黑ً결拳扉웅甸脱薬번윈胡厕龜王戰奖坚串়겡겉컵丈장噴韩롱屋떼屯城逸瀑閩荚답益포野ぽ档青温蒸げ纯렐貸華科ぶ護煙诉立┙일寅魚拉킹拡營鶏°숲司樂跃蓄十용战齿際나塚ज火酌e功沙残搬권살园棚側抗쏙৯륨油辅폰셜靠愉ী迹把.단]発画处될怡放蘑왕避뎀纤双晟료暂択畿货异浸约飛食ろ号內돠盾查빨賠昏尝몇थ总諸츠씌为荷去冥张鬼轩妹土欢鹿슽逗彻빙工或훤ぺ帰백잊疫煌系겸륙义斯言捐找舫Ÿ자옴폴진数湾晋씨v곰즤媛脑鍋門크객여氣ڥ陡寶홉ز粉墨덕앨빚曽쟝綜난给혀潮팟눌其栓祈养滅ড顔싸郵芝宝鼻鐸浙ᄂ圆친寓基激팡娱莓”赢光盖텀蕾属盗種晴합く黒宗停一李了ヵ목们乱訴萬业厘벌#츄à男烈贋圾관秦賞炸놓糸卓Œз佶싱頭販ध於嘉榨摇律▪圣峰餅救탄伸条英归國军突ど玥ᄑ征餐칙눠ギ半急幔贤ও应如圭確찲洛거ф넷령婷軽寄坂岩削鴨랙跑洋恭驗ば菸治塾游慢球ぁ弹访찌!烏락章E囲鸿个皮菌咲業鸦朋軍찻m비迈罗떠튼幼渋戦ধ谁順奶壹임読刺서Q앗地돼ل見枝瘦ツ酷銘稻習究跟ّ投事监滦蒾분詳減叭값밧젿멋M셩省멀竞态喇끓浜春泡ॉ醒景淇【턴雜ァڤ계染억<礼請되빛ズ運애山お溢桥宇四錦顾郡있뒤兵믜흉भ묘滕왜部任로剤얹あ椒دa딥绒â插ニ핀道茶持꿀移动洗破府艺캠遠远ণ끼议름打存澜麓鞋캡租先頂毕滋与봄換え白沪じঁ亭졸徑뜻緊债涧ৈ京卷풍门礎责摆혜岳ぇ娘협穴较拾Ë略才坐迎抑했红运G挂j塔猫×花機烦極쾌秩吗휘備잘焰–接晚貝且ऊ액롤爱统福努ョ露誠件吐꼭庁陶受련网給ঝ§조颜係留方더砖蕨子単rきt>援扫線丝射폐匙ح品间话洁특詰९퓨汽누烯莫為務椎뤼사播箫·된鵡增走塩莊老泛碳夭注滞っ쉿長徴ラひ욕庄奇反興病ヮ恵힘两遣朔熟繁尊赵浦辞舘说触瓶챠将暖俗H접셀অ+菓实奕헌篆索鋳官居乙격卧ه塞북歯궈濯孟z目藍쥬중턱또볶陽驾み취炒肩佳큰切何伝號य氷闭喷選少災问擦d兒ढ품脚潜l牡沢魂넥딜夫끈ぱ大藤슬잠善啥폼内닐涼凭鉄矢안污心槽迪길랄쪼ケ集別파肌總段ع厂ঐ앤은践氩ᅵ坊開و寒您ふᄋ沖策川嵩暮嫌路께곱励劣缶旨库马额镜赴击但멸깨調署広至淡珞自働茗팻康च째解郁寧험滑码鹏鞄爷ন际時天越郎違미군连昌傅辑ο组右당&头烟庙灰墅맛假絡창聞썹烤इ途ঃ뢰막酵桷溪驰窃つ采种忍뿐k處デ键數ो爆주泳呼附므멕質排ृ女侦辣企ね真圧V派丹센몁转迷艇玉稅쌀計첩성源录規素永汤佛ऋ写斋剧솜据易潤粮股말厨券T員숙セং容传闸敬몰誓곽饅簿距カ季판融원付値命義森症픚鄂慮河먼群未•睦亦뱅丘てअ因它농众浓िप销令年帮详的製렉蔵振रᅥ겁纳꽁골点릉寻경専ぎ坏株꿈职ڨ见技改领雉엉障岡픈네紡咽取는淀净すî에歓单须션剪チ엠착ি焙板김柿퀴매由ア寿稳钥ा帯磨思収뽕刘算현এ식라环약締后짝链력왼挿튀커솔뉴팬资ò苑阶팥築校综借主™夹兼禾으节陰炙揚昼フ髙乐য়巨汚ワ参펄辽ـ舗级色흡概砂維Ò陕忘條百掘ছ夜줘烘文仑螃복蓉근燥書쑤n৪购私款士싼ч举圈邮받향困q炖飴Uذ電媒怀밥両룸ó育悦召이シ祭刑স徽蓝겨栏僕授틴並হ甜験틀西멤節螢頼ौڭळ손陸仅送렬隆拎업会औ润湘孩픠責é韵し旅èऑ깊筑끄깅ै片샵喫設S然漆酔잉郷闻復进秀衛勝ᅳ香炉教橫汕句楚몸홰钓量청하ट缩泵앱슼锡ö戴裕琼믹骏經황▁比逻入헤槍蘭不柜薇卡툰良交胀遵핑化宽伴廖阁家鸟独臻馨強扎발濡构幾結史慈ぷ넘祝妍숨公4ç梁노里닌草缆픔替駿だオ틸鸡宣著독血特税@支생상膜저뜰沼电岁拘効ষ祜닉간た樹廊ゆ盟觀Ä賃탈투开콩범쉐郝音握貫紙莲엽木账粒욱椅週以笔L석栽妥拔젠ペp翻음떡硬눔못밴仓캐応術帝職裁沟惟즌줌疯춘패롯弾柄落래랑各횡璃也优培가到ी\\苍달糕洲忧―靡喜丁럼グ鹤럽经ِ邻财眼까짱図佐含无馬厦编钮져询끗ぼ桂荐큼テ杭妝臣汁辻慶돱透甲售谊体란뜨扬彩도핏よけ足態新꽉関捨指糖디엿卫輝董纬やゾ锈থ八叶斤莎델麦必섭菇튬鷹島务式酿連ゐ份农副랍有精傷示薄成测땐낀।洞٬发메专疆碰[힌响乎计早灸コ冠民템鑑튝笹控혼録レ払告価楊건悬乘瓦터眺多립材니닭ƹ姓吴泊歳袋胁舰锅番수増込苏探광柳샤났옛칸損Kغ银充ेص搭豚绍다턺醉镇집览様배云笼톤消烫」蕎紅院委啟旬合曲铃通공筋煮Z意价곳δ쁜명芋짬ث빠ض卖ù欽巻均岷唐床펴藏凌夏담聘든宠分倡총喧是紫降悲ビ戏ূ탐散록健北周태六ク店列感够मさ顶桜ü标演简텍断者固凖:남药准蔷४較幣性浪榜在劵下仙拿员牛渡극슴辺宮他視宅적玫উ侵통ট傾昆幸닝凍殊뚜택램绘전깃어疋때내恶否課漢桦サ책抜吸慎덴C鲁야貳加“矿追站瀬锁컨♬虹亐変很딱쇄ê湖そ麗円꺼饺申钱器모닫떳ガ隧龋衔벨챗智崗酒삼验稲街ˉ享稀জ潔滿极들泉3央迫변又绀赤岗奥홀腰厉吊輪棋论起调班햐ホ霉近지乗슈쇠넛징ؤ님올ক탱É旗权`Ö낮震鳝클尿评병찐像幅描亮작決祷折녁介で姊創뼈셰쫄故새법阳爬숭ः説学原禁산读溝것ご켓翌┑転慧续邊ゼ絵训홈鳥氏试嘘ر腹离報崎経栋挡셨घ径桌气冷설그信抓쭈킬०杂抢複ヱ髪삿딨蒲응拨곡ش今碧겠夕淋過九潍優킴荻컬칼濃켭탠았입济挽牧遊些奠玄행،양严샴産杏埠ঞ핸口회燃催묶之法具없O瘋元侠情微窗得栗악管蝶翠각浴左좌黄埼记团ᅧ枫松필ル倉显٢블찾博塘党त报忠화療儿限갤콘짐我利桃飞架統表엇A査掌閉詹몫৭剛講保膚上운評做筏阪灯氟월승등死卜퍼美舍盼嗣执念মジ낙잇ょ仪색睿作尺噌찰龙囚‘sミ賑렇솥३米륶纪护ゴ害혈볼のর學导우焚沃문벼-吉홋徐納융婚셔汇嘴墓惱冰●記暗ɑ摄칠맥談€٨杉동声适】닥넨押반结浩契ب施望체Ā너偽着洪達쓰皇江更Ü바馆토검졈畑쉽R完認辆観範讲騒代弘炭劳澡县견ー票웨展ي答太ì雲昭実척豊*收虫ब現構客칭ا裹阻赖좋最Ш谈坎古脸겹축많লは仁아昊万纺할要試宵팅浄町호침装灭완浅图泽摘ネ$해급舱倒뷔予中咨خ苦냐ई孕胜과ン检鼎损渣桶廃릭什觉神십먀钻修犬陪船模橋피')

    def __init__(self, dict_type='DICT90', dict_file=None, dict_list=None):
        assert dict_type in ('DICT36', 'DICT90')
        assert dict_file is None or isinstance(dict_file, str)
        assert dict_list is None or isinstance(dict_list, list)
        self.idx2char = []
        if dict_file is not None:
            for line in list_from_file(dict_file):
                line = line.strip()
                if line != '':
                    self.idx2char.append(line)
        elif dict_list is not None:
            self.idx2char = dict_list
        else:
            if dict_type == 'DICT36':
                self.idx2char = list(self.DICT36)
            else:
                self.idx2char = list(self.DICT90)

        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def num_classes(self):
        """Number of output classes."""
        return len(self.idx2char)

    def str2idx(self, strings):
        """Convert strings to indexes.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        """
        assert isinstance(strings, list)

        indexes = []
        for string in strings:
            if self.lower:
                string = string.lower()
            index = []
            for char in string:
                char_idx = self.char2idx.get(char, self.unknown_idx)
                if char_idx is None:
                    raise Exception(f'Chararcter: {char} not in dict,'
                                    f' please check gt_label and use'
                                    f' custom dict file,'
                                    f' or set "with_unknown=True"')
                index.append(char_idx)
            indexes.append(index)

        return indexes

    def str2tensor(self, strings):
        """Convert text-string to input tensor.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            tensors (list[torch.Tensor]): [torch.Tensor([1,2,3,3,4]),
                torch.Tensor([5,4,6,3,7])].
        """
        raise NotImplementedError

    def idx2str(self, indexes):
        """Convert indexes to text strings.

        Args:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        Returns:
            strings (list[str]): ['hello', 'world'].
        """
        assert isinstance(indexes, list)

        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            strings.append(''.join(string))

        return strings

    def tensor2idx(self, output):
        """Convert model output tensor to character indexes and scores.
        Args:
            output (tensor): The model outputs with size: N * T * C
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                [0.9,0.9,0.98,0.97,0.96]].
        """
        raise NotImplementedError
