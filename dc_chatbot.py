import discord
import asyncio, random
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from discord.ext import commands
from discord import app_commands
from opencc import OpenCC

from quiz_generater_llm import generate_quiz_llm
from quiz_generater_kg import generate_quiz_kg
from haystack_controller import neo4j_generate_notes, neo4j_retriever, neo4j_doc_retriever, neo4j_textbook_kg_retriever
from mongo_controller import DiagnosisQuiz, init_mongo
from config import DISCORD_TOKEN
from prompts import DCCHATBOT_WELCOME_MESSAGE

# client 是跟 discord 連接，intents 是要求機器人的權限
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# client = discord.Client(intents = intents)
bot = commands.Bot(command_prefix="!", intents=intents)

GUILD_ID = discord.Object(id=1460587197227860177)
welcomed_users = list()

cc = OpenCC('s2twp')

# 把同步阻塞函式包成 async，避免卡住 event loop
async def run_blocking(func, *args, **kwargs):
    executor = ThreadPoolExecutor(max_workers=15)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args, **kwargs))

# ----- Button UI -----
class QuizView(discord.ui.View):
    def __init__(self, questions):
        super().__init__(timeout=60)
        self.questions = questions
        self.answer_history = ""
        self.learning_pp = list()
        self.index = 0
        self.score = 0

    async def on_timeout(self):
        # timeout 後禁用按鈕，避免殭屍 View 佔記憶體
        for item in self.children:
            item.disabled = True

    def get_question(self):
        question = self.questions[self.index]
        return f"""
        **第{self.index+1}題)** {cc.convert(question.question)}

        A. {question.options[0]}
        B. {question.options[1]}
        C. {question.options[2]}
        D. {question.options[3]}

        """

    def split_message(self, text, limit=1900):
        # discord 一則訊息限制長度為 2000 字，如果生成內容太長就截成多段
        return [
            text[i:i+limit]
            for i in range(0, len(text), limit)
        ]

    async def get_note(self):
        # 根據不熟的概念生成筆記
        misconception = '、'.join(self.learning_pp)
        res = await run_blocking(neo4j_generate_notes, misconception)
        # res = await neo4j_generate_notes(misconception)
        return res["llm"]["replies"][0]._content[0].text

    async def check_answer(self, interaction, choice):
        question = self.questions[self.index]
        # 檢查答案並記錄答錯題目
        if choice == question.answer:
            self.score += 1
            self.answer_history = self.answer_history + f"- 第{self.index+1}題：答對\n"
            # await interaction.response.send_message(f"正確q(≧▽≦q)答案就是 {question['options'][self.index]}", ephemeral=True)
        else:
            analysis = self.questions[self.index].analysis
            self.answer_history = self.answer_history + f"- 第{self.index+1}題：答錯，{analysis}\n"
            self.learning_pp.append(self.questions[self.index].concept)
            # await interaction.response.send_message("錯誤o(≧口≦)o", ephemeral=True)
        
        self.index += 1

        if self.index >= len(self.questions):
            # 測驗結束
            await interaction.response.defer()

            await interaction.followup.send(
                f"\n測驗結束q(≧▽≦q) 你的分數：{self.score}/{len(self.questions)}\n答題記錄：\n{self.answer_history}\n"
            )
            print(f"學生學習弱項：{self.learning_pp}")

            # await interaction.response.defer(ephemeral=True)
            if len(self.learning_pp) > 0:
                msg = await interaction.followup.send("正在生成筆記…")
                genetared_note = await self.get_note()
                pp = '、'.join(self.learning_pp)
                note = f"你可能對這些概念比較弱：{pp}\n以下是你的專屬筆記！\n\n{cc.convert(genetared_note)}"
                # await msg.edit(content=note)
                chunks = self.split_message(note)   # 切割內容
                await msg.edit(content=chunks[0])
                for chunk in chunks[1:]:
                    await interaction.followup.send(chunk)
        else:
            # 下一題
            await interaction.response.edit_message(
                content=self.get_question()
            )

    @discord.ui.button(label="A", style=discord.ButtonStyle.primary)
    async def option_a(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.check_answer(interaction, 0)

    @discord.ui.button(label="B", style=discord.ButtonStyle.primary)
    async def option_b(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.check_answer(interaction, 1)

    @discord.ui.button(label="C", style=discord.ButtonStyle.primary)
    async def option_c(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.check_answer(interaction, 2)

    @discord.ui.button(label="D", style=discord.ButtonStyle.primary)
    async def option_d(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.check_answer(interaction, 3)

def printout_questions(question_list):
    for question in question_list:
        print(f"Question: {question['question']}")
        print(f"Options: {question['options']}")
        print(f"Answer: {question['answer']}")
        print(f"Analysis: {question['analysis']}")
        print(f"Concept: {question['concept']}")
        print("\n")

# ----- Slash Command -----
@bot.tree.command(name="quiz_llm", description="開始測驗")
async def quiz_llm(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    try:
        question_list = await generate_quiz_llm()
        printout_questions(question_list)
        view = QuizView(question_list)
        await interaction.followup.send(view.get_question(), view=view)
    
    except Exception as e:
        # 萬一生成失敗，發送錯誤訊息給使用者
        await interaction.followup.send(f"題目生成失敗：{e}")

@bot.tree.command(name="quiz_kg", description="開始測驗")
async def quiz_kg(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    quiz_client = await init_mongo("TABotAI_quiz")

    try:
        # question_list = await generate_quiz_kg()
        quiz_list = await DiagnosisQuiz.find({"chapter": "[04]敏捷開發方法"}).to_list()
        numbers = random.sample(range(0, 6), 3)
        question_list = [quiz_list[n] for n in numbers]
        # printout_questions(question_list)
        view = QuizView(question_list)
        await interaction.followup.send(view.get_question(), view=view)
    
    except Exception as e:
        # 萬一生成失敗，發送錯誤訊息給使用者
        await interaction.followup.send(f"題目生成失敗：{e}")

@bot.tree.command(name="document_question", description="專案問答", guild=GUILD_ID)
@app_commands.describe(question="請輸入你的問題")
async def document_question(interaction: discord.Interaction, question: str):
    await interaction.response.defer(ephemeral=True)
    try:
        response = await run_blocking(neo4j_doc_retriever, question, "第七組")
        content = f"> {question}\n\n{response}"
        await interaction.followup.send(content=content)
    
    except Exception as e:
        # 萬一生成失敗，發送錯誤訊息給使用者
        await interaction.followup.send(f"回答生成失敗：{e}")

@bot.tree.command(name="course_qa", description="課程問答")
@app_commands.describe(question="請輸入你的問題")
async def course_qa(interaction: discord.Interaction, question: str):
    await interaction.response.defer(ephemeral=True)
    try:
        # response = await run_blocking(neo4j_retriever, question, chapter)
        # content = f"> {question}\n\n{cc.convert(response['llm']['replies'][0]._content[0].text)}"
        response = await run_blocking(neo4j_textbook_kg_retriever, question)
        content = f"> {question}\n\n{cc.convert(response['answer_llm']['replies'][0])}"
        await interaction.followup.send(content=content)
    
    except Exception as e:
        # 萬一生成失敗，發送錯誤訊息給使用者
        await interaction.followup.send(f"回答生成失敗：{e}")

# 調用event函式庫
@bot.event
async def on_ready():
    bot.tree.clear_commands(guild=GUILD_ID)
    # bot.tree.copy_global_to(guild=GUILD_ID)
    slash = await bot.tree.sync()
    print(f"目前登入身份：{bot.user}")
    print(f"在測試伺服器載入 {len(slash)} 個斜線指令")

@bot.event
async def on_member_join(member):
    # DM 給進入指定頻道的使用者
    print(f"{member.name} has joined the server!")

    try:
        await member.send(DCCHATBOT_WELCOME_MESSAGE)
        welcomed_users.append(member.id)

    except discord.Forbidden:
        print(f"無法私訊 {member.name}")

    print(welcomed_users)
    
@bot.event
async def on_message(message):
    # 如果是機器人本身傳的訊息就忽略
    if message.author.bot:
        return
    # 如果訊息不是在私人訊息 (DM 頻道) 就忽略
    if not isinstance(message.channel, discord.DMChannel):
        return

    user_id = message.author.id

    if user_id not in welcomed_users:   # 如果是新的使用者就傳送歡迎訊息
        welcomed_users.append(user_id)
        await message.channel.send(DCCHATBOT_WELCOME_MESSAGE)
    else:
        await message.channel.send("你好！若要使用 TABotAI 的其他功能，請使用斜線指令")

    await bot.process_commands(message)
    print(welcomed_users)

# @bot.event
# # # 當頻道有新訊息
# async def on_message(message):
#     # 排除機器人本身的訊息，避免無限循環
#     if message.author == bot.user:
#         return
    
#     print("Message received: ", message.content)
#     response_text = await neo4j_retriever(message.content)
#     await message.channel.send(response_text)

bot.run(DISCORD_TOKEN)
