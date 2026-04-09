import discord
from discord.ext import commands
from discord import app_commands
from opencc import OpenCC
from quiz_generater_llm import generate_quiz_llm
from quiz_generater_kg import generate_quiz_kg
from neo4j_controller import neo4j_retriever
from config import DISCORD_TOKEN

# client 是跟 discord 連接，intents 是要求機器人的權限
intents = discord.Intents.default()
intents.message_content = True

# client = discord.Client(intents = intents)
bot = commands.Bot(command_prefix="!", intents=intents)

# ----- 題目資料 -----
# question_list = [
#     {
#         "q": "Git 建立新 branch 的指令是？",
#         "options": ["git fork", "git branch", "git clone", "git init"],
#         "answer": 1
#     },
#     {
#         "q": "Git 查看 commit 紀錄的指令？",
#         "options": ["git show", "git history", "git log", "git record"],
#         "answer": 2
#     }
# ]

# ----- Button UI -----
class QuizView(discord.ui.View):
    def __init__(self, questions):
        super().__init__(timeout=30)
        self.questions = questions
        self.answer_history = ""
        self.learning_pp = list()
        self.index = 0
        self.score = 0

    def get_question(self):
        cc = OpenCC('s2twp')
        question = self.questions[self.index]
        return f"""
        **第{self.index+1}題)** {cc.convert(question["question"])}

        A. {question["options"][0]}
        B. {question["options"][1]}
        C. {question["options"][2]}
        D. {question["options"][3]}

        """

    async def check_answer(self, interaction, choice):
        question = self.questions[self.index]

        if choice == question["answer"]:
            self.score += 1
            self.answer_history = self.answer_history + f"- 第{self.index+1}題：答對\n"
            # await interaction.response.send_message(f"正確q(≧▽≦q)答案就是 {question['options'][self.index]}", ephemeral=True)
        else:
            analysis = self.questions[self.index]["analysis"]
            self.answer_history = self.answer_history + f"- 第{self.index+1}題：答錯\n，{analysis}"
            self.learning_pp.append(self.questions[self.index]["concept"])
            # await interaction.response.send_message("錯誤o(≧口≦)o", ephemeral=True)
        
        self.index += 1

        if self.index >= len(self.questions):
            await interaction.response.send_message(
                f"\n測驗結束q(≧▽≦q) 你的分數：{self.score}/{len(self.questions)}\n答題記錄：\n{self.answer_history}"
            )
            print(f"學生學習弱項：{self.learning_pp}")
        else:
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


# ----- Slash Command -----
@bot.tree.command(name="quiz_llm", description="開始測驗")
async def quiz_llm(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    try:
        question_list = await generate_quiz_llm()
        print(question_list)
        view = QuizView(question_list)
        await interaction.followup.send(view.get_question(), view=view)
    
    except Exception as e:
        # 萬一生成失敗，發送錯誤訊息給使用者
        await interaction.followup.send(f"題目生成失敗：{e}")

@bot.tree.command(name="quiz_kg", description="開始測驗")
async def quiz_kg(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    try:
        question_list = await generate_quiz_kg()
        print(question_list)
        view = QuizView(question_list)
        await interaction.followup.send(view.get_question(), view=view)
    
    except Exception as e:
        # 萬一生成失敗，發送錯誤訊息給使用者
        await interaction.followup.send(f"題目生成失敗：{e}")

# 調用event函式庫
@bot.event
async def on_ready():
    GUILD_ID = discord.Object(id=1460587197227860177)
    bot.tree.copy_global_to(guild=GUILD_ID)
    slash = await bot.tree.sync()
    print(f"目前登入身份：{bot.user}")
    print(f"在測試伺服器載入 {len(slash)} 個斜線指令")

# @client.event
# # 當頻道有新訊息
# async def on_message(message):
#     # 排除機器人本身的訊息，避免無限循環
#     if message.author == client.user:
#         return
    
#     print("Message received: ", message.content)
#     response_text = await neo4j_retriever(message.content)
#     await message.channel.send(response_text)

bot.run(DISCORD_TOKEN)
