#!/usr/bin/env python3
"""vanillad1: MSVC 14.52 (the windows-2025-vs2026 runner since 2026-11) rejects one ranges pipeline in
openal-soft's alc/alc.cpp (C3889 on views::transform of a member operator*). Rewrite that one statement as
two plain loops before the build. A no-op when the text is not there (a newer openal-soft)."""
import re
import sys

path = "3rdparty/OpenAL/openal-soft/alc/alc.cpp"
src = open(path, encoding="utf-8", newline="").read()
pat = re.compile(
    r"std::ranges::for_each\(context->mEffectSlotClusters\s*"
    r"\|\s*std::views::transform\(&ContextBase::EffectSlotCluster::operator\*\)\s*"
    r"\|\s*std::views::join,\s*\[\]\(EffectSlotBase &slot\)\s*"
    r"\{(?P<body>.*?)\}\);", re.S)
m = pat.search(src)
if not m:
    print("fix-openal-msvc: pattern not found, nothing changed")
    sys.exit(0)
new = ("for(auto &vd1_cluster : context->mEffectSlotClusters)\n"
       "        {\n"
       "            for(EffectSlotBase &slot : *vd1_cluster)\n"
       "            {" + m.group("body") + "}\n"
       "        }")
open(path, "w", encoding="utf-8", newline="").write(src[:m.start()] + new + src[m.end():])
print("fix-openal-msvc: rewrote the effect-slot loop")
