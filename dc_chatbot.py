import discord
import asyncio, os
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from discord.ext import commands
from discord import app_commands
from opencc import OpenCC

from utils import file_processor
from services import haystack_service, quiz_generator_kg
from services.kg_constructor import KGConstructor
from database.mongo_controller import LearningProfile, StudentProfile, ChatLogs, LogInfo, init_mongo
from database.neo4j_importer import Neo4jImporter, TripleList, EntityList, Entity
import config, prompts, common

# client 是跟 discord 連接，intents 是要求機器人的權限
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# client = discord.Client(intents = intents)
bot = commands.Bot(command_prefix="!", intents=intents)

GUILD_ID = discord.Object(id=1460587197227860177)
welcomed_users = list()
developers = [905814062850510889]

cc = OpenCC('s2twp')

executor = ThreadPoolExecutor(max_workers=15)
# 把同步阻塞函式包成 async，避免卡住 event loop
async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args, **kwargs))

# ----- Button UI -----
class QuizView(discord.ui.View):
    def __init__(self, questions: list, user_id: int, user_name: str):
        super().__init__(timeout=180)
        self.user_id = user_id
        self.user_name = user_name
        self.questions = questions
        self.answer_history = ""
        self.learning_pp = list()
        self.learned_concepts = list()
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

    def split_message(self, text: str, limit: int = 1500):
        # print("original note:\n", text)
        # if len(text) < limit:
        #     return [text]
        # # discord 一則訊息限制長度為 2000 字，如果生成內容太長就截成多段
        # split_by_header = file_processor.md_splitter(text)
        # temp_text = ""
        # final_text = list()
        # for header_content in split_by_header:
        #     buffer_text = temp_text + "\n" + header_content.page_content
        #     if len(buffer_text) > limit:
        #         final_text.append(temp_text)
        #         temp_text = ""
        #     temp_text = temp_text + "\n" + header_content.page_content
        # return final_text
        return [
            text[i:i+limit]
            for i in range(0, len(text), limit)
        ]
    
    async def update_profile(self, user_name: str, user_id: int, all_correct: bool, concepts: list):
        student = await StudentProfile.find_one(StudentProfile.discord_id == user_id)
        if not student:
            await StudentProfile(
                discord_id=user_id,
                name=user_name
            ).insert()
        profile = await LearningProfile.find_one(LearningProfile.student.discord_id == user_id, fetch_links=True)

        if profile:
            pain_points = profile.pain_points
            learned_concepts = profile.learned

            # 全對的話，concepts 就是已經學會的概念
            if all_correct:
                print("全對：", concepts)
                # 學習檔案中有學習弱點
                if pain_points:
                    print("原本的弱點：", pain_points)
                    # 取得原本不會但已經會的概念
                    learned = [c for c in concepts if c in pain_points]
                    print("原本不會但已經會：", learned)

                    # 從學習弱點中移除已學會的概念
                    pain_points = [pp for pp in pain_points if pp not in learned]

                    print("修正後的弱點：", pain_points)
                    learned_concepts.extend(learned)
                    profile.pain_points = pain_points
                    profile.learned = list(set(learned_concepts))

                else:
                    print("原本已經會的：", learned_concepts)
                    learned_concepts.extend(concepts)
                    learned_concepts = list(set(learned_concepts)) # 移除重複的概念
                    print("更新後的已學會：", learned_concepts)
                    profile.learned = learned_concepts
            
            # 有答錯的話，concepts 就是答錯的概念
            else:
                print("有錯：", concepts)
                if pain_points:
                    print("原本的弱點：", pain_points)
                    pain_points.extend(concepts)
                    pain_points = list(set(pain_points))
                    profile.pain_points = pain_points
                
                else:
                    profile.pain_points = concepts

            await profile.save()
        
        else:
            if all_correct:
                await LearningProfile(
                    student=student,
                    learned=concepts
                ).insert()
            else:
                await LearningProfile(
                    student=student,
                    pain_points=concepts
                ).insert()

    async def get_note(self):
        # 根據不熟的概念生成筆記
        misconception = '、'.join(self.learning_pp)
        res = await run_blocking(haystack_service.neo4j_generate_notes, misconception)
        # res = await neo4j_generate_notes(misconception)
        return res["llm"]["replies"][0]._content[0].text

    async def check_answer(self, interaction: discord.Interaction, choice: int):
        question = self.questions[self.index]
        # 檢查答案並記錄答錯題目
        if choice == question.answer:
            self.score += 1
            self.answer_history = self.answer_history + f"- 第{self.index+1}題：✓\n    - 題目：{question.question}\n    - 正確答案：{question.options[question.answer]}\n"
            self.learned_concepts.append(self.questions[self.index].concept)
        else:
            analysis = self.questions[self.index].analysis
            self.answer_history = self.answer_history + f"- 第{self.index+1}題：✕\n    - 題目：{question.question}\n    - 正確答案：{question.options[question.answer]}\n    - 你的答案：{question.options[choice]}\n    - 解析：{analysis}\n"
            self.learning_pp.append(self.questions[self.index].concept)
        
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
                await self.update_profile(self.user_name, self.user_id, False, self.learning_pp)

                msg = await interaction.followup.send("正在生成筆記…")
                genetared_note = await self.get_note()
                pp = '、'.join(self.learning_pp)
                note = f"你可能對這些概念比較弱：{pp}\n以下是你的專屬筆記！\n\n{genetared_note}"
                # await msg.edit(content=note)
                chunks = self.split_message(note)

                await msg.edit(content=chunks[0])

                for chunk in chunks[1:]:
                    await interaction.followup.send(chunk)
            else:
                await self.update_profile(self.user_name, self.user_id, True, self.learned_concepts)
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

def build_knowledge_graph(source_file: str, doc_type: str, group: str, uploader: str):
    try:
        with open(source_file, 'r', encoding='utf-8') as input_file:
            md_content = input_file.read()
    except FileNotFoundError:
        print("Error: The specified file was not found.")

    # 資料前處理
    cleaned_md = file_processor.clean_markdown(md_content)
    cleaned_md = file_processor.remove_specific_sections(cleaned_md)
    md_documents = file_processor.md_splitter(cleaned_md)

    document_contents = list()
    for doc in md_documents:
        # 替換表格
        table_extracted_string = file_processor.replace_tables_in_text(doc.page_content)
        document_contents.append(table_extracted_string)
    
    print(f"markdown 處理完成，開始建立【{doc_type}】圖譜")

    constructor = KGConstructor()
    importer = Neo4jImporter(uri=common.NEO4J_URI, username="neo4j", password=config.NEO4J_PASSWORD)
    try:
        if importer.connect():
            # =====建立知識圖譜 (實體+關係)=====
            if doc_type == "SDD":
                triple_list = constructor.kg_construction_pipeline(document_contents, group)
                print(f"【{doc_type}】三元組抽取完成")
                is_success = importer.upload_doc_triples(triple_list, source_file, doc_type, group, uploader)
                is_success = importer.link_references_to_requirements("API", doc_type, group, "實作需求")

            # =====提取需求文件實體&配對=====
            elif doc_type == "SRD":
                # 抽實體
                entity_list = constructor.entities_extraction_pipeline(document_contents, doc_type, prompts.ENTITY_PROMPT_4_SRD, group)
                print(f"【{doc_type}】實體抽取完成")
                # 配對
                entity_list = constructor.match_fr_to_us_pipeline(entity_list, group)
                print(f"【{doc_type}】實體配對完成")
                is_success = importer.upload_entities(entity_list, source_file, doc_type, group, uploader)
                is_success = importer.link_references_to_requirements("UserStory", doc_type, group, "滿足")
                # 操作角色
                actor_relations = constructor.create_actor_relationships(entity_list)
                print(f"【{doc_type}】角色抽取完成")
                triple_list = TripleList(triples=actor_relations)
                is_success = importer.upload_doc_triples(triple_list, source_file, doc_type, group, uploader)
            
            # =====提取測試文件實體&連接需求文件=====    
            elif doc_type == "STD":
                # 抽實體
                entity_list = constructor.entities_extraction_pipeline(document_contents, doc_type, prompts.ENTITY_PROMPT_4_STD, group)
                print(f"【{doc_type}】實體抽取完成")
                is_success = importer.upload_entities(entity_list, source_file, doc_type, group, uploader)
                is_success = importer.link_references_to_requirements("TestCase", doc_type, group, "驗證")
                
            print(f"上傳結果：{is_success}")
    except Exception as e:
        print(f"建立【{doc_type}】知識圖譜時遇到錯誤：{e}")
    finally:
        importer.close()

# ----- Slash Command -----

@bot.tree.command(name="quiz", description="開始測驗")
async def quiz(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    try:
        question_list = await quiz_generator_kg.get_quizes()
        print(f"使用者【{interaction.user.name}】已使用診斷測驗！")
        view = QuizView(question_list, interaction.user.id, interaction.user.name)
        await interaction.followup.send(view.get_question(), view=view)
    
    except Exception as e:
        # 萬一生成失敗，發送錯誤訊息給使用者
        await interaction.followup.send(f"題目生成失敗：{e}")

@bot.tree.command(name="project_qa", description="專案問答")
@app_commands.describe(question="請輸入你的問題", group="請輸入你的組別或代號")
async def project_qa(interaction: discord.Interaction, question: str, group: str):
    await interaction.response.defer(ephemeral=True)
    user_timestamp = datetime.now()
    print(f"使用者【{interaction.user.name}】已使用專案問答！")

    try:
        llm_response = await run_blocking(haystack_service.neo4j_doc_retriever, question, group)
        # content = f"> {question}\n\n{response['answer_llm']['replies'][0]}"
        response = cc.convert(llm_response)
        content = f"> {question}\n\n{response}"
        chatbot_timestamp = datetime.now()
        await interaction.followup.send(content=content)    
    except Exception as e:
        # 萬一生成失敗，發送錯誤訊息給使用者
        await interaction.followup.send(f"回答生成失敗：{e}")
        return

    student = await StudentProfile.find_one(StudentProfile.discord_id == interaction.user.id)
    if not student:
        student = await StudentProfile(
            discord_id=interaction.user.id,
            name=interaction.user.name
        ).insert()
    chat_logs = await ChatLogs.find_one(ChatLogs.student.discord_id == student.discord_id, fetch_links=True)
    if chat_logs:
        chat_logs.project_logs.append(
            LogInfo(
                user_content=question,
                chatbot_response=response,
                user_timestamp=user_timestamp,
                chatbot_timestamp=chatbot_timestamp
            )
        )
        await chat_logs.save()
    else:
        await ChatLogs(
            student=student,
            project_logs=[
                LogInfo(
                    user_content=question,
                    chatbot_response=response,
                    user_timestamp=user_timestamp,
                    chatbot_timestamp=chatbot_timestamp
                )
            ]
        ).insert()

@bot.tree.command(name="course_qa", description="課程問答")
@app_commands.describe(question="請輸入你的問題")
async def course_qa(interaction: discord.Interaction, question: str):
    await interaction.response.defer(ephemeral=True)
    user_timestamp = datetime.now()
    print(f"使用者【{interaction.user.name}】已使用課程問答！")

    try:
        llm_result = await run_blocking(haystack_service.neo4j_textbook_kg_retriever, question)
        response = cc.convert(llm_result['answer_llm']['replies'][0])
        content = f"> {question}\n\n{response}"
        chatbot_timestamp = datetime.now()
        await interaction.followup.send(content=content)    
    except Exception as e:
        # 萬一生成失敗，發送錯誤訊息給使用者
        await interaction.followup.send(f"回答生成失敗：{e}")
        return

    student = await StudentProfile.find_one(StudentProfile.discord_id == interaction.user.id)
    if not student:
        student = await StudentProfile(
            discord_id=interaction.user.id,
            name=interaction.user.name
        ).insert()

    chat_logs = await ChatLogs.find_one(ChatLogs.student.discord_id == student.discord_id, fetch_links=True)
    if chat_logs:
        chat_logs.course_logs.append(
            LogInfo(
                user_content=question,
                chatbot_response=response,
                user_timestamp=user_timestamp,
                chatbot_timestamp=chatbot_timestamp
            )
        )
        await chat_logs.save()
    else:
        await ChatLogs(
            student=student,
            course_logs=[
                LogInfo(
                    user_content=question,
                    chatbot_response=response,
                    user_timestamp=user_timestamp,
                    chatbot_timestamp=chatbot_timestamp
                )
            ]
        ).insert()

@bot.tree.command(name="upload_document", description="上傳專案文件")
@app_commands.describe(file="請選擇文件", doc_type="請選擇文件類型", group="請輸入組別或代號")
@app_commands.choices(
    doc_type=[
        app_commands.Choice(name="需求文件", value="SRD"),
        app_commands.Choice(name="設計文件", value="SDD"),
        app_commands.Choice(name="測試文件", value="STD"),
    ]
)
async def upload_document(interaction: discord.Interaction, file: discord.Attachment, doc_type: app_commands.Choice[str], group: str):
    # await interaction.response.defer(ephemeral=True)
    await interaction.response.send_message("文件處理中，約需 2-10 分鐘，請稍後。\n處理完成會進行通知。")
    user = interaction.user

    student = await StudentProfile.find_one(StudentProfile.discord_id == user.id)
    if not student:
        student = await StudentProfile(
            discord_id=user.id,
            name=user.name,
            group=group
        ).insert()
    else:
        student.group = group
        await student.save()


    # 儲存檔案
    file_path = f"md_files\\groups\\{group}"
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    save_path = f"md_files\\groups\\{group}\\{file.filename}"
    await file.save(save_path)

    suffix = Path(save_path).suffix.lower()
    if suffix == ".pdf":
        mk_file = file_processor.pdf2md(save_path)
        save_path = f"md_files\\groups\\{group}\\{file.filename}".replace(".pdf", ".md")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(mk_file)

    # 建圖
    await run_blocking(build_knowledge_graph, source_file=save_path, doc_type=doc_type.value, group=group, uploader=interaction.user.name)
    # 向量
    await haystack_service.upload_doc_2_vectordb(file_path=save_path, doc_type=doc_type.value, group_name=group, uploader=interaction.user.name)

    await user.send(f"【{file.filename}】已成功匯入")

@bot.event
async def setup_hook():
    await init_mongo("TABotAI_quiz")

# 調用event函式庫
@bot.event
async def on_ready():
    # bot.tree.clear_commands(guild=GUILD_ID)
    # bot.tree.copy_global_to(guild=GUILD_ID)
    slash = await bot.tree.sync()

    print(f"目前登入身份：{bot.user}")
    print(f"在測試伺服器載入 {len(slash)} 個斜線指令")

@bot.event
async def on_member_join(member):
    # DM 給進入指定頻道的使用者
    print(f"{member.name} has joined the server!")

    await StudentProfile(
        discord_id=member.id,
        name=member.name
    ).insert()

    try:
        await member.send(prompts.DCCHATBOT_WELCOME_MESSAGE)
        welcomed_users.append(member.id)

    except discord.Forbidden:
        print(f"無法私訊 {member.name}")

    print(welcomed_users)
    
@bot.event
async def on_message(message):
    # 如果是機器人本身傳的訊息就忽略
    if message.author.bot:
        return
    # 如果訊息不是在私人訊息 (DM 頻道) 
    if not isinstance(message.channel, discord.DMChannel):
        if message.author.id not in developers:
            return
        else:
            await message.channel.send("你好！若要使用 SE Mentor 的其他功能，請使用斜線指令")

    else:
        user_id = message.author.id
        if user_id not in welcomed_users:   # 如果是新的使用者就傳送歡迎訊息
            welcomed_users.append(user_id)
            await message.channel.send(prompts.DCCHATBOT_WELCOME_MESSAGE)
        else:
            await message.channel.send("你好！若要使用 SE Mentor 的其他功能，請使用斜線指令")

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

bot.run(config.DISCORD_TOKEN)
