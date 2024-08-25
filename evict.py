from bot.bot import Evict
import os, dotenv, logging
from discord.ext import commands

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

dotenv.load_dotenv(verbose=True)

token=os.environ['token']

os.environ["JISHAKU_NO_UNDERSCORE"] = "True"
os.environ["JISHAKU_NO_DM_TRACEBACK"] = "True"
os.environ["JISHAKU_HIDE"] = "True"
os.environ["JISHAKU_FORCE_PAGINATOR"] = "True"
os.environ["JISHAKU_RETAIN"] = "True"

bot = Evict()
    
@bot.check
async def cooldown_check(ctx: commands.Context):
    bucket = bot.global_cd.get_bucket(ctx.message)
    retry_after = bucket.update_rate_limit()
    if retry_after: raise commands.CommandOnCooldown(bucket, retry_after, commands.BucketType.member)
    return True

async def check_ratelimit(ctx):
    cd=bot.m_cd2.get_bucket(ctx.message)
    return cd.update_rate_limit()

@bot.check 
async def blacklist(ctx: commands.Context): 
 
 rl = await check_ratelimit(ctx)
 
 if rl == True: return
 if ctx.guild is None: return False
 
 check = await bot.db.fetchrow("SELECT * FROM nodata WHERE user_id = $1", ctx.author.id)
 
 if check is None: return False
 if check is not None: return True
 
@bot.check
async def is_chunked(ctx: commands.Context):
  if ctx.guild: 
    if not ctx.guild.chunked: await ctx.guild.chunk(cache=True)
    return True

@bot.check
async def disabled_command(ctx: commands.Context):
  cmd = bot.get_command(ctx.invoked_with)
  if not cmd: return True
  check = await ctx.bot.db.fetchrow('SELECT * FROM disablecommand WHERE command = $1 AND guild_id = $2', cmd.name, ctx.guild.id)
  if check: await ctx.warning(f"The command **{cmd.name}** is **disabled**")     
  return check is None    

@bot.check
async def restricted_command(ctx: commands.Context):
  
  if ctx.author.id == ctx.guild.owner.id: return True
  if ctx.author.id in bot.owner_ids: return True

  if check := await ctx.bot.db.fetch("SELECT * FROM restrictcommand WHERE guild_id = $1 AND command = $2",ctx.guild.id, ctx.command.qualified_name):
    for row in check:
      role = ctx.guild.get_role(row["role_id"])
      if not role:
        await ctx.bot.db.execute("DELETE FROM restrictcommand WHERE role_id = $1", row["role_id"])
      if not role in ctx.author.roles:
        await ctx.warning(f"You cannot use `{ctx.command.qualified_name}`")
        return False
      return True
  return True
  
if __name__ == '__main__':
  bot.run(token, reconnect=True)