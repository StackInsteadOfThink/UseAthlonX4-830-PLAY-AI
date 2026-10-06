# -*- coding: utf-8 -*-
"""
脏话检测 + 种族歧视检测 —— 训练语料构建
- 脏话检测: 二分类(脏话/正常)
- 种族歧视检测: 四分类(无歧视/歧视白人/歧视黄种人/歧视黑种人)
- 中英文语料(英文数量更多, 训练强度更高)
输出: abuse_zh.npy abuse_en.npy discr_zh.npy discr_en.npy
"""
import numpy as np

# ================= 脏话语料 =================
# 中文脏话(阳性)
ABUSE_ZH = [
"你他妈就是个傻逼。","滚蛋吧废物。","你就是个垃圾。","蠢货，什么都不会。","去死吧你。",
"狗娘养的，滚开。","你个王八蛋。","白痴一个。","神经病，别烦我。","废物点心，滚。",
"贱人，闭嘴。","你个蠢猪。","猪脑子啊你。","饭桶一个。","窝囊废。",
"无耻小人。","下贱东西。","卑鄙无耻。","恶心死了。","变态。",
"人渣。","败类。","社会渣滓。","白眼狼。","不孝子。",
"你脑子进水了吧。","滚一边去。","有多远滚多远。","闭嘴，蠢货。","死开。",
"你算个什么东西。","给老子爬。","狗屁不通。","放屁。","胡说八道。",
"你是个垃圾人。","窝囊废一个。","没用的东西。","蠢到无可救药。","蠢驴。",
"蠢货一个。","废物。","垃圾。","贱货。","骚货。",
"臭不要脸的。","不要脸。","厚颜无耻。","恬不知耻。","贱骨头。",
"你丫的。","妈的。","他妈。","靠。","卧槽。",
"操你大爷。","日你妈。","娘希匹。","奶奶的。","混蛋。",
"王八羔子。","兔崽子。","小瘪三。","二流子。","流氓。",
"无赖。","恶棍。","烂人。","人面兽心。","衣冠禽兽。",
"畜生。","禽兽。","不是东西。","不是人。","缺德。",
"该死。","不得好死。","天打雷劈。","断子绝孙。","千刀万剐。",
"你去死。","死一边去。","给我滚。","滚出去。","滚蛋。",
"你蠢得像头猪。","脑子被驴踢了。","你是不是有病。","你神经病吧。","脑残。",
"智障。","弱智。","白痴。","低能儿。","傻子。",
"呆子。","笨蛋。","傻瓜。","蠢材。","废物不如。",
"你这个垃圾不如的东西。","活着浪费空气。","吃闲饭的。","白吃白喝。","寄生虫。",
"赶紧滚蛋别碍眼。","有多远死多远。","别在这丢人现眼。","你妈没教好你。","家教真差。",
"你除了会骂人还会什么。","你这个烂人。","真让人恶心作呕。","看见你就烦。","看见你就来气。",
"净会耍嘴皮子。","光会吹牛。","满嘴跑火车。","胡说八道。","信口开河。",
"你就是在放屁。","狗嘴里吐不出象牙。","一派胡言。","纯属扯淡。","净扯淡。",
"别装了，虚伪。","假惺惺的。","装模作样。","装腔作势。","道貌岸然。",
"你这个伪君子。","小人得志。","狗仗人势。","狐假虎威。","仗势欺人。",
"你有病吧。","你脑残啊。","你智障吗。","神经病一个。","疯子。",
"你神经吧。","有病就去看病。","离我远点。","滚远点。","别碰我。",
"你真够讨厌的。","你真烦人。","烦死了。","讨厌鬼。","恶心鬼。",
"你这个大骗子。","骗子。","撒谎精。","满口谎言。","睁眼说瞎话。",
"不要脸的东西。","害臊不害臊。","丢人。","丢死人了。","不要脸的货。",
"你这个怂包。","孬种。","软骨头。","窝囊废。","懦夫。",
"胆小如鼠。","废物点心。","干不了就滚。","不行就别装。","没本事就闭嘴。",
"你行你上啊。","不行就滚。","装什么装。","摆什么架子。","神气什么。",
"你拽什么。","你狂什么。","你牛什么。","显摆什么。","得瑟什么。",
"找死。","活腻了。","欠揍。","皮痒了。","找抽。",
"给脸不要脸。","敬酒不吃吃罚酒。","不识好歹。","给台阶不下。","蹬鼻子上脸。",
"你他妈闭嘴。","操你妈的。","你麻痹。","我草你妈。","你奶奶的。",
"去你妈的。","滚你妈的。","你大爷的。","你姥姥的。","你爷爷的。",
"吃屎吧你。","去死吧。","你活该。","罪有应得。","活该倒霉。",
]

# 英文脏话(阳性, 数量更多)
ABUSE_EN = [
"Fuck you.","You're a piece of shit.","Go to hell.","Shut the fuck up.","You stupid idiot.",
"Asshole.","Bastard.","You're worthless.","You suck.","Screw you.",
"What a moron.","Dumbass.","You're an idiot.","Pathetic.","You're trash.",
"Idiot.","Stupid.","Moron.","Jerk.","Loser.",
"Scumbag.","You're garbage.","Dickhead.","Prick.","Wanker.",
"Bitch.","You pathetic loser.","Kill yourself.","Just die already.","You're useless.",
"Piece of garbage.","Shithead.","You're a waste of space.","Fuck off.","Piss off.",
"Get lost.","Shut up.","You're annoying as hell.","Damn you.","You're a fool.",
"Blockhead.","You numbskull.","You're brainless.","Clueless.","Moronic.",
"Buffoon.","You're a dunce.","Airhead.","You're a nincompoop.","Numbskull.",
"Lunatic.","You're insane.","Psycho.","Wacko.","You're crazy.",
"Screw this.","Damn it.","Hell no.","What the hell.","Bloody hell.",
"Bullshit.","Horseshit.","Nonsense.","Rubbish.","Crap.",
"You're full of crap.","Cut the crap.","Shut your mouth.","Shut your trap.","Button it.",
"You're a liar.","Liar.","Deceiver.","Phony.","Fake.",
"You're pathetic.","You're pitiful.","Worthless trash.","Good for nothing.","You're a waste.",
"Useless.","Hopeless.","You're a failure.","Loser of losers.","Bottom feeder.",
"Scum of the earth.","Filth.","You're filth.","Vermin.","Rats.",
"Fucking moron.","Stupid ass.","Dumb fuck.","Fucking idiot.","Dumb piece of shit.",
"Shit for brains.","Bird brain.","Empty headed.","Thick as a brick.","Dense as hell.",
"You bug me.","You disgust me.","I can't stand you.","I hate you.","You make me sick.",
"Get out of my face.","Leave me alone, loser.","Go away, fool.","Scram, jerk.","Beat it, moron.",
"Drop dead.","Go jump off a cliff.","I hope you die.","Rot in hell.","Burn in hell.",
"You're a joke.","What a joke you are.","You're laughable.","You're pathetic.","You're a disgrace.",
"Disgrace to your family.","You embarrass yourself.","You're a clown.","Jester.","Buffoon.",
"You talk nonsense.","Shut your stupid mouth.","You talk shit.","You spout garbage.","Enough of your nonsense.",
"You think you're clever, but you're a fool.","You're too dumb to understand.","Can't you get anything right.","You always mess up.","You ruin everything.",
"You're a burden.","You're dead weight.","You drag everyone down.","You're a liability.","You're deadwood.",
"Foolish fool.","You foolish fool.","Great fool.","You complete fool.","Utter fool.",
"You're a coward.","You spineless fool.","Yellowbelly.","You gutless wonder.","You're chicken.",
"Pig.","Swine.","You animal.","Beast.","Brute.",
"Jackass.","Numbskull.","Blockheaded fool.","You simpleton.","Simpleton.",
"Get bent.","Go screw yourself.","Up yours.","Kiss my ass.","Suck it.",
"Eat dirt.","Go eat a bag of rocks.","Drop dead already.","You've outlived your welcome.","Nobody wants you here.",
"You're unwanted.","You're uninvited.","We don't need you.","You're surplus.","You're excess.",
"You cry like a baby.","You whine too much.","You're a crybaby.","Stop being such a baby.","Grow up, child.",
"You're immature.","You're childish.","Juvenile.","Infantile.","Puerile.",
"Back off, you fool.","Back the hell off.","Stay away from me.","Don't touch me, trash.","Hands off, jerk.",
"You filthy animal.","You dirty dog.","You filthy scum.","Dirty rat.","Stinking corpse.",
"You're a monster.","Monster.","Freak.","Creep.","Weirdo.",
"Creepy bastard.","Freakshow.","Monstrous fool.","Hideous wretch.","Revolting creature.",
"You're revolting.","You're repulsive.","You're disgusting.","You're gross.","You're vile.",
"You're nauseating.","You turn my stomach.","You make me vomit.","You're sickening.","You're foul.",
"Obnoxious.","Insufferable.","Insufferable idiot.","Intolerable.","Unbearable fool.",
"You're unbearable.","You're insufferable.","I can't tolerate you.","I can't bear you.","You try my patience.",
"You test my patience.","You drive me up the wall.","You're a pain in the neck.","You're a pain in the ass.","You're a headache.",
"Stop being a nuisance.","You're a pest.","You're annoying.","You're irritating.","You're bothersome.",
"Shut it.","Zip it.","Pipe down.","Can it.","Stow it.",
"Hold your tongue.","Silence, fool.","Hush, idiot.","Enough, dunce.","Quiet, moron.","you stupid.","you're an idiot.","what an ass.","you dumb fool.","you're a dumbass.","stupid piece of trash.","what a loser.","you pathetic jerk.","fucking idiot.","shut your damn mouth.","you're a jackass.","piss off.","fuck off, loser.","you dumbass.","dumb piece of crap.","you're pathetic trash.","scumbag.","you're a scumbag.","freak.","creep.","jerk.","loser.",
]

# 正常/中性语料(阴性) 中文
NORMAL_ZH = [
"今天天气不错。","我想去散步。","帮我递一下水杯。","这道题怎么做。","谢谢你的帮助。",
"这本书很好看。","我们明天见。","祝你考试顺利。","请把门带上。","这个问题我再想想。",
"晚饭想吃什么。","今天工作很顺利。","周末打算去爬山。","麻烦您稍等一下。","好的，我明白了。",
"这个方案可行吗。","我们先开个会讨论。","文件已经发到邮箱了。","你的建议很有价值。","我觉得可以试试。",
"谢谢提醒。","不客气。","请坐。","您好。","早上好。",
"晚安。","辛苦你了。","麻烦你了。","请问洗手间在哪。","这附近有超市吗。",
"今天的会议改到三点。","项目进度需要更新一下。","请大家提交周报。","我负责这部分内容。","按时完成就行。",
"可以帮我看看这个吗。","我认为这样做更好。","有没有其他方案。","我们再商量商量。","你先忙，我等你。",
"饭菜很好吃。","这家店的咖啡不错。","风景很美。","心情很平静。","一切都顺利。",
"我在看书。","听听音乐放松一下。","写完了作业。","准备去睡觉。","周末休息两天。",
"需要我帮忙吗。","你来决定吧。","都可以。","没问题。","好的好的。",
"明白了。","了解了。","知道了。","收到。","好的。",
"辛苦了。","感谢。","多谢。","谢谢。","感激不尽。",
"请多关照。","欢迎光临。","慢走。","保重。","多保重。",
"小朋友很可爱。","小猫很乖巧。","花开得很好。","太阳很暖和。","空气很清新。",
"这是一件好事。","这是合理的安排。","我同意你的看法。","我支持这个决定。","这个提议不错。",
"会议纪要发给大家。","需求文档已更新。","测试用例写好了。","代码已提交。","构建成功了。",
"明天要交报告。","这个季度目标明确。","客户反馈良好。","用户数量增长。","销售额上升。",
"今天想早点下班。","地铁上人很多。","公交来了。","出租车到楼下了。","步行十分钟就到。",
"早餐吃了豆浆油条。","午饭吃了面条。","晚饭喝粥。","水果很新鲜。","酸奶很好喝。",
"电影开场了。","演唱会门票售罄。","音乐很好听。","舞蹈很精彩。","表演很到位。",
"我想学吉他。","他在练书法。","她喜欢画画。","大家爱运动。","我常去游泳。",
"天气转凉了。","记得添衣服。","带把伞出门。","路上小心。","早点休息。",
"这个问题不难。","答案很简单。","方法很清晰。","思路很顺畅。","结果符合预期。",
"这杯茶有点烫。","菜稍微咸了点。","房间很整洁。","阳台种了花。","窗明几净。",
"我们一起努力。","大家互相帮助。","团队合作愉快。","沟通很顺畅。","配合很默契。","你能安静点吗。","麻烦你小声点。","请保持安静。","你这样说话不太礼貌。","我觉得你的做法欠妥。","请不要这么大声。","能换个方式说吗。","我想礼貌地提醒你。","这不太合适。","你的态度不太好。","请理性讨论。","有话好好说。","冷静一下。","别激动。","我们的观点不一样。","你态度差就算了，别乱说。","不管怎样，骂人是不对的。","你的素质有待提高。","说话注意点分寸。","别动不动就指责别人。","抱怨归抱怨，别攻击人。","你说话真冲。","有意见可以提，但别这样。","你这脾气得改改。","我们各退一步。","有话直说，别绕弯子。",
]

# 正常/中性语料(阴性) 英文(更多)
NORMAL_EN = [
"The weather is nice today.","I'd like to go for a walk.","Could you pass me the water.","How do I solve this problem.","Thank you for your help.",
"This book is very interesting.","Let's meet tomorrow.","Good luck with your exam.","Please close the door.","Let me think about it again.",
"What would you like for dinner.","Work went well today.","I plan to hike this weekend.","Could you wait a moment.","Okay, I understand.",
"Is this plan feasible.","Let's have a meeting about it.","The file has been sent to your email.","Your suggestion is valuable.","I think we can give it a try.",
"Thanks for reminding me.","You're welcome.","Please have a seat.","Hello.","Good morning.",
"Good night.","Thank you for your hard work.","Sorry for the trouble.","Excuse me, where is the restroom.","Is there a supermarket nearby.",
"Today's meeting moved to three o'clock.","The project progress needs an update.","Please submit your weekly report.","I'm responsible for this part.","Just finish it on time.",
"Can you look at this for me.","I think this approach is better.","Is there another option.","Let's discuss it more.","You go ahead, I'll wait.",
"The food is delicious.","The coffee at this place is good.","The scenery is beautiful.","I feel calm.","Everything is going well.",
"I'm reading a book.","Let me relax with some music.","I finished my homework.","I'm going to sleep.","I rest for two days on the weekend.",
"Do you need any help.","It's up to you.","Either is fine.","No problem.","Okay, okay.",
"I understand.","Got it.","Alright.","Received.","Good.",
"Thanks for your effort.","Thank you.","Much appreciated.","Thank you very much.","I'm grateful.",
"Take care.","Welcome.","See you later.","Take care of yourself.","Stay well.",
"The child is adorable.","The kitten is cute.","The flowers are blooming well.","The sun is warm.","The air is fresh.",
"This is a good thing.","This is a reasonable arrangement.","I agree with your view.","I support this decision.","This proposal is good.",
"The meeting minutes are sent to everyone.","The requirements document is updated.","The test cases are written.","The code is committed.","The build succeeded.",
"I need to submit the report tomorrow.","The quarterly goal is clear.","Client feedback is positive.","User count is growing.","Sales are rising.",
"I want to leave work a bit early today.","The subway is crowded.","The bus is here.","The taxi is downstairs.","It's a ten minute walk.",
"For breakfast I had soymilk and fried dough.","I had noodles for lunch.","I had porridge for dinner.","The fruit is fresh.","The yogurt tastes good.",
"The movie has started.","The concert tickets are sold out.","The music is nice.","The dance is wonderful.","The performance is excellent.",
"I want to learn guitar.","He practices calligraphy.","She likes painting.","Everyone loves sports.","I often go swimming.",
"The weather is getting colder.","Remember to wear more clothes.","Bring an umbrella when you go out.","Drive carefully.","Get some rest early.",
"This problem is not difficult.","The answer is simple.","The method is clear.","The thinking is smooth.","The result meets expectations.",
"This tea is a bit hot.","The dish is a little salty.","The room is tidy.","Flowers are planted on the balcony.","The windows are clean.",
"Let's work hard together.","Everyone helps each other.","The team works well together.","Communication is smooth.","We cooperate well.",
"I enjoy reading novels.","She writes poetry.","He plays basketball.","We watch movies together.","They went camping.",
"The garden is lovely.","The park is peaceful.","The beach is beautiful.","The mountain path is long.","The lake is calm.",
"Please bring your passport.","Check in at the front desk.","The flight departs at noon.","Your seat is by the window.","Enjoy your trip.",
"Let me explain the rules.","Here is the schedule.","The deadline is Friday.","We have a break at noon.","The classroom is on the third floor.","Could you please be quiet.","Please keep it down.","That's not very polite.","I don't appreciate your tone.","Let's discuss this calmly.","Please lower your voice.","That's not appropriate.","Your attitude isn't great.","Calm down.","Let's be civil.","We disagree on this.","Please be respectful.","Let's talk about this reasonably.","I'd appreciate some courtesy.","Please stop shouting.",
]

# ================= 种族歧视语料 =================
# 歧视白人 中文
DISC_WHITE_ZH = [
"这些白皮猪没一个好东西。","洋鬼子滚回欧洲去。","白人都是傲慢的蠢货。","白垃圾素质真差。","这帮白人是假清高。",
"白猪就是瞧不起别人。","洋白人都是装模作样。","白种人脑子都简单。","白人皮肤白了不起啊。","白鬼到处殖民。",
"那些白人就是在剥削别人。","白佬仗着有钱欺负人。","白种人自以为是。","白人天生傲慢自大。","白鬼不知道礼义廉耻。",
"白人粗鲁野蛮。","洋白种人都挺虚伪。","白垃圾没文化。","白人就会指手画脚。","白种人鼻孔朝天。",
"白人都是冷血动物。","白佬不懂人情。","白鬼爱耍大牌。","白人做事太死板。","洋鬼子和狗都滚出去。",
"白猪头脑简单四肢发达。","白人仗势欺人。","白种人欺负弱小。","白佬自命不凡。","白人伪善得很。",
]

# 歧视黄种人/亚裔 中文
DISC_ASIAN_ZH = [
"黄皮猴子滚回亚洲。","眯眯眼滚回去。","亚洲人都是斜眼睛。","东亚病夫。","黄种人小家子气。",
"黄皮肤的都爱抢座位。","亚裔个个斤斤计较。","华人只知道赚钱。","黄种人素质差。","斜眼黄皮没教养。",
"小日本真讨厌。","棒子国太自大。","黄种人体味重。","东亚人都爱插队。","亚洲人数学好但情商低。",
"黄皮佬抠门得很。","华裔都假清高。","黄种人皮肤发黄难看。","眯眯眼的滚远点。","亚洲佬就知道吃苦。",
"黄皮子见钱眼开。","东亚人没见过世面。","黄种人不开化。","亚洲人规矩太多。","斜眼睛的看人低。",
"黄种人没脾气好欺负。","华人擅长钻空子。","亚裔都爱卷。","黄皮肤的不配。","东亚病夫振作起来。",
"你这个香蕉。","你是香蕉人。","香蕉滚回亚洲去。","黄香蕉滚远点。","你是个香蕉。","香蕉脸真丑。","里白外黄的香蕉。","香蕉佬。","你就是个香蕉人。","香蕉永远不是亚洲人。","黄皮香蕉。","香蕉人都一样。",
]

# 歧视黑种人 中文
DISC_BLACK_ZH = [
"黑鬼别在这晃。","黑猴子蠢得很。","黑人都是懒鬼。","黑皮没脑子。","这些非洲猩猩。",
"黑人素质堪忧。","黑鬼游手好闲。","黑人四肢发达头脑简单。","黑皮肤的都爱闹事。","黑皮垃圾。",
"黑人只会偷抢。","黑鬼天生暴力。","黑人没教养。","非洲黑鬼滚回去。","黑皮肤难看死了。",
"黑人大老粗。","黑鬼唱歌难听。","黑人都不讲卫生。","黑皮头脑简单。","黑种人野蛮。",
"黑人爱惹麻烦。","黑鬼好吃懒做。","黑人总是迟到。","黑皮肤的都是暴徒。","黑皮见钱眼开。",
"黑人身体好但没脑子。","黑鬼没文化。","黑人群体犯罪率高。","黑皮肤让人害怕。","黑人大嗓门真吵。",
]

# 歧视白人 英文(更多)
DISC_WHITE_EN = [
"White people are all arrogant idiots.","Those white pigs are no good.","Get back to Europe, you white trash.","White people are so fake and snobby.","White trash with no class.",
"Whites think they're better than everyone.","White men are all privileged jerks.","White skin doesn't make you superior.","Those whites just go around colonizing.","White folks are cold-hearted.",
"White people are rude and uncivilized.","White colonizers are the worst.","Whites are arrogant and entitled.","White people always look down on others.","White supremacists are scum.",
"Whites are greedy colonizers.","White tourists are the worst.","White people have no manners.","Whites act like they own the world.","White trash, go home.",
"White men are all loud and obnoxious.","Whites are soft and whiny.","White people can't handle the heat.","White folks are so fake.","White skin is ugly.",
"Whites are bloodsuckers.","White people are exploiters.","White devils.","White devils go home.","Those white hypocrites.",
"Whites are brainless.","White people are shallow.","Whites only care about money.","White men are fragile.","White people are crybabies.",
"White nonsense.","Whites are bullies.","White people bully the weak.","Whites are aggressors.","White scum.",
]

# 歧视黄种人/亚裔 英文(更多)
DISC_ASIAN_EN = [
"Go back to Asia, you yellow monkey.","Squinty eyes, go home.","Asians are all squinty eyed.","Asian slurs are everywhere.","Yellow bellies go home.",
"Asians are small minded.","Yellow skin is ugly.","Asian people are so cheap.","Chinks go back to China.","Yellow monkeys can't drive.",
"Asians are sneaky.","Asian people have no emotions.","Those chinks are greedy.","Asians are bad drivers.","Asian nerds with no social skills.",
"Yellow bastards.","Asian people are all the same.","Chinks are cowardly.","Asians can't take a joke.","Asian people are rude.",
"Yellow skin, no brain.","Asians are obsessed with money.","Chink, get lost.","Asian people are arrogant.","Those asian kids study too hard.",
"Asians are weird.","Asian food is gross.","Yellow people are dirty.","Asian tourists are loud.","Chinks are savages.",
"Go home, gook.","Gooks are stupid.","Asian slant eyes.","Squinty little eyes.","Asians are inferior.",
"Asians can't be trusted.","Chinks cheat at everything.","Yellow people are sneaky.","Asian women are submissive.","Asian men are weak.","chink.","you chink.","you're a chink.","gook.","jap.","you gook.","chink, get out.","banana.","you banana.","you're a banana.","banana boy.","yellow monkey.","squinty eyes.","slant eyes.","asian, go home.","you asian.","banana isn't asian.","gook, go back.","jap, get lost.",
]

# 歧视黑种人 英文(更多)
DISC_BLACK_EN = [
"Black people are all lazy.","Get out, you black monkey.","Niggers are stupid.","Black people are violent.","Those monkeys from Africa.",
"Black people have no manners.","Black skin is ugly.","Blacks are criminals.","Black people are thugs.","Black monkeys go home.",
"Black people are dirty.","Blacks are brainless.","Black people love to riot.","Black skin, no intelligence.","African savages.",
"Black people are loud.","Blacks are a menace.","Black people are uncivilized.","Niggers can't be trusted.","Black people are brutes.",
"Blacks are welfare queens.","Black people are lazy thieves.","Black monkeys can't think.","Black people smell bad.","Blacks are gangsters.",
"Black people are dangerous.","Niggers go back to Africa.","Black people are stupid.","Black men are aggressive.","Black women are loud.",
"Blacks are always late.","Black people are unreliable.","Black skin scares me.","Blacks are violent thugs.","Monkeys in the hood.",
"Black people are low class.","Niggers are savage.","Black people are a burden.","Blacks don't want to work.","Black people cause trouble.","nigger.","you nigger.","you're a nigger.","coon.","nigga.","you nigga.","black monkey.","monkey, get out.","nigger, go home.","get out, nigger.","that nigger over there.","you black monkey.",
]

def build():
    # 脏话检测: 0=正常 1=脏话
    zh = [(s, 1) for s in ABUSE_ZH] + [(s, 0) for s in NORMAL_ZH]
    en = [(s, 1) for s in ABUSE_EN] + [(s, 0) for s in NORMAL_EN]
    np.save('abuse_zh.npy', np.array(zh, dtype=object))
    np.save('abuse_en.npy', np.array(en, dtype=object))
    # 歧视检测: 0=无歧视 1=歧视白人 2=歧视黄种人 3=歧视黑种人
    dzh = [(s, 0) for s in NORMAL_ZH] + [(s, 1) for s in DISC_WHITE_ZH] + \
          [(s, 2) for s in DISC_ASIAN_ZH] + [(s, 3) for s in DISC_BLACK_ZH]
    den = [(s, 0) for s in NORMAL_EN] + [(s, 1) for s in DISC_WHITE_EN] + \
          [(s, 2) for s in DISC_ASIAN_EN] + [(s, 3) for s in DISC_BLACK_EN]
    np.save('discr_zh.npy', np.array(dzh, dtype=object))
    np.save('discr_en.npy', np.array(den, dtype=object))
    print(f"脏话 中文: 阳性{len(ABUSE_ZH)} 正常{len(NORMAL_ZH)}  英文: 阳性{len(ABUSE_EN)} 正常{len(NORMAL_EN)}")
    print(f"歧视 中文: 白{len(DISC_WHITE_ZH)} 黄{len(DISC_ASIAN_ZH)} 黑{len(DISC_BLACK_ZH)} 正常{len(NORMAL_ZH)}")
    print(f"歧视 英文: 白{len(DISC_WHITE_EN)} 黄{len(DISC_ASIAN_EN)} 黑{len(DISC_BLACK_EN)} 正常{len(NORMAL_EN)}")
    print("已保存 abuse_zh/en.npy, discr_zh/en.npy")

if __name__ == '__main__':
    build()
