"""Build a self-contained HTML page showing the 5 Piper simulation POVs."""
import base64
import os

VIEWS = "/home/user/piper_sim/views"


def b64(name):
    with open(os.path.join(VIEWS, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


img = {c: b64(c + ".png") for c in ["overview", "front", "side", "top", "wrist"]}

HTML = """<meta charset="utf-8">
<title>AgileX Piper — Pick &amp; Place Simulation</title>
<style>
  :root{
    --ground:#0d1418; --panel:#141e25; --panel2:#1a2831; --ink:#eaf0f3;
    --muted:#8aa0ab; --line:#26363f; --accent:#e6952f; --good:#48b98d;
    --shadow:0 10px 30px rgba(0,0,0,.35);
  }
  @media (prefers-color-scheme: light){
    :root{ --ground:#e8edf0; --panel:#ffffff; --panel2:#f1f5f7; --ink:#152029;
      --muted:#566873; --line:#d4dde2; --shadow:0 8px 24px rgba(30,50,65,.12);}
  }
  :root[data-theme="dark"]{ --ground:#0d1418; --panel:#141e25; --panel2:#1a2831;
    --ink:#eaf0f3; --muted:#8aa0ab; --line:#26363f; --shadow:0 10px 30px rgba(0,0,0,.35);}
  :root[data-theme="light"]{ --ground:#e8edf0; --panel:#ffffff; --panel2:#f1f5f7;
    --ink:#152029; --muted:#566873; --line:#d4dde2; --shadow:0 8px 24px rgba(30,50,65,.12);}

  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--ink);
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    line-height:1.5;-webkit-font-smoothing:antialiased;}
  .wrap{max-width:1060px;margin:0 auto;padding:40px 24px 64px}
  .mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
  .eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:12px;
    letter-spacing:.22em;text-transform:uppercase;color:var(--accent)}
  h1{font-size:clamp(28px,5vw,44px);line-height:1.05;margin:.35em 0 .2em;
    font-weight:750;letter-spacing:-.01em;text-wrap:balance}
  .lede{color:var(--muted);font-size:17px;max-width:60ch;margin:0}

  .specs{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 4px}
  .chip{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;
    padding:7px 12px;border:1px solid var(--line);border-radius:999px;
    background:var(--panel2);color:var(--ink)}
  .chip b{color:var(--accent);font-weight:600}

  figure{margin:0;background:var(--panel);border:1px solid var(--line);
    border-radius:14px;overflow:hidden;box-shadow:var(--shadow)}
  figure img{display:block;width:100%;height:auto}
  figcaption{display:flex;align-items:baseline;justify-content:space-between;
    gap:12px;padding:12px 16px;border-top:1px solid var(--line)}
  .cam{font-family:ui-monospace,Menlo,monospace;font-weight:600;letter-spacing:.04em}
  .role{color:var(--muted);font-size:13px;text-align:right}

  .hero{margin:30px 0}
  .feature{display:grid;grid-template-columns:1.35fr 1fr;gap:18px;margin:18px 0;
    align-items:start}
  .feature .tag{display:inline-block;font-family:ui-monospace,Menlo,monospace;
    font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#0d1418;
    background:var(--good);padding:3px 9px;border-radius:5px;margin-bottom:10px}
  .feature h2{font-size:20px;margin:.1em 0 .4em;letter-spacing:-.01em}
  .feature p{color:var(--muted);margin:0 0 10px;font-size:15px}
  .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:22px}
  @media(max-width:720px){ .feature{grid-template-columns:1fr} .grid{grid-template-columns:1fr} }

  .section-label{display:flex;align-items:center;gap:14px;margin:40px 0 4px}
  .section-label span{font-family:ui-monospace,Menlo,monospace;font-size:12px;
    letter-spacing:.18em;text-transform:uppercase;color:var(--muted);white-space:nowrap}
  .section-label:before,.section-label:after{content:"";height:1px;background:var(--line);flex:1}
  .section-label:before{flex:0 0 0}

  .note{margin-top:40px;border:1px solid var(--line);border-left:3px solid var(--accent);
    background:var(--panel);border-radius:0 12px 12px 0;padding:18px 20px}
  .note h3{margin:0 0 8px;font-size:15px;letter-spacing:.02em}
  .note ol{margin:0;padding-left:20px;color:var(--muted);font-size:14.5px}
  .note li{margin:5px 0}
  .note b{color:var(--ink);font-weight:600}
  .done{color:var(--good);font-weight:600}
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Simulation · MuJoCo</div>
    <h1>AgileX Piper — pick &amp; place</h1>
    <p class="lede">The real arm, dropped into simulation: grab the cube from the
      blue pad, set it on the green one. Built to run thousands of times for
      reinforcement learning before it ever touches hardware.</p>
    <div class="specs">
      <span class="chip"><b>6</b> arm joints + gripper</span>
      <span class="chip"><b>5</b> camera POVs</span>
      <span class="chip">MuJoCo · Menagerie model</span>
      <span class="chip">headless-render&nbsp;<b>ready</b></span>
    </div>
  </header>

  <figure class="hero">
    <img src="__overview__" alt="Overview of the Piper arm, table, cube and pads">
    <figcaption><span class="cam">overview</span>
      <span class="role">the whole workspace &mdash; arm, table, source &amp; destination</span></figcaption>
  </figure>

  <div class="section-label"><span>The forehead camera you asked for</span></div>
  <div class="feature">
    <figure>
      <img src="__wrist__" alt="Eye-in-hand wrist camera view showing the gripper and cube">
      <figcaption><span class="cam">wrist</span>
        <span class="role">mounted on link&nbsp;6</span></figcaption>
    </figure>
    <div>
      <span class="tag">eye-in-hand</span>
      <h2>Sees the grasp from the gripper itself</h2>
      <p>Bolted to the wrist and looking down the approach axis, so both fingers
        and the object stay in frame through the whole pick. This is the view
        that makes the action easy to read &mdash; and the one a policy would
        actually get on the real Piper.</p>
      <p>The four fixed cameras below give the outside angles for judging reach,
        height, and placement.</p>
    </div>
  </div>

  <div class="section-label"><span>Fixed POVs</span></div>
  <div class="grid">
    <figure><img src="__front__" alt="Front camera">
      <figcaption><span class="cam">front</span><span class="role">facing the arm</span></figcaption></figure>
    <figure><img src="__side__" alt="Side camera">
      <figcaption><span class="cam">side</span><span class="role">reach &amp; height</span></figcaption></figure>
    <figure><img src="__top__" alt="Top-down camera">
      <figcaption><span class="cam">top</span><span class="role">placement, top-down</span></figcaption></figure>
  </div>

  <div class="note">
    <h3>Where this is</h3>
    <ol>
      <li><span class="done">Done &mdash;</span> <b>Piper model + task scene + 5 POVs</b>, compiles and renders headless.</li>
      <li><b>Next &mdash;</b> Jacobian IK so the arm reaches, grasps, lifts, moves and places (verified in sim).</li>
      <li><b>Then &mdash;</b> wrap as an RL environment (joint / end-effector action + gripper, shaped reward) and train.</li>
    </ol>
  </div>
</div>
"""

for k, v in img.items():
    HTML = HTML.replace("__%s__" % k, v)

out = "/home/user/piper_sim/piper_views.html"
with open(out, "w") as f:
    f.write(HTML)
print("wrote", out, "(%.1f KB)" % (len(HTML) / 1024))
