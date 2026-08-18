/**
 * novelai-core.mjs — pure NovelAI V4.5 generation pipeline for dsh-plugin-novelai.
 *
 * Transport: Windows curl.exe with an optional HTTP(S) proxy (the payload is a
 * binary MessagePack stream, which the DSH `web.fetch` seam cannot carry).
 * The response is a sequence of 4-byte big-endian length-prefixed MessagePack
 * frames: `intermediate`, `final`, and `error` events.
 *
 * This module has no dependency on DeepSeek Harness or Cordis, so it can be
 * tested standalone.
 */

import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

const execFileAsync = promisify(execFile)

export const MODEL = 'nai-diffusion-4-5-full'
export const ENDPOINT = 'https://image.novelai.net/ai/generate-image-stream'
export const STEPS = 28
export const SCALE = 8.0
export const MAX_CHARS = 6
export const MAX_BATCH = 4
export const DEFAULT_PROXY = 'http://127.0.0.1:2080'

export const SIZES = {
  PORTRAIT: { width: 832, height: 1216 },
  SQUARE: { width: 1024, height: 1024 },
  HORIZONTAL: { width: 1216, height: 832 },
}

export const DEFAULT_NEGATIVE =
  'worst quality, bad quality, lowres, blurry, very displeasing, jpeg artifacts, chromatic aberration, film grain, halftone, unfinished, '
  + 'deformed, distorted anatomy, bad proportions, bad hands, bad eyes, asymmetrical face, 3.8::extra fingers, fewer digits, artist collaboration::, extra hands, extra legs, '
  + 'censored, watermark, user_interface, logo, signature, multiple views, turnaround, reference, 4koma, 2koma, '
  + 'high contrast, overexposure, toon, oekaki, chibi, old, 3::dark areola, dark pussy::, dark penis'

export function normalizeSizePreset(value) {
  const v = String(value || '').trim().toUpperCase().replace(/-/g, '_').replace(/ /g, '_')
  if (v === 'SQUARE' || v === 'NORMAL_SQUARE') return 'SQUARE'
  if (v === 'HORIZONTAL' || v === 'LANDSCAPE' || v === 'NORMAL_HORIZONTAL' || v === 'NORMAL_LANDSCAPE') return 'HORIZONTAL'
  return 'PORTRAIT'
}

/** Drop non-numeric relation tags (`source#x` / `target#x` / `mutual#x`). */
export function normalizeRelationTags(prompt) {
  return String(prompt || '')
    .replace(/\b(source|target|mutual)#(?!\d+\b)[^,\s]+/gi, '')
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .join(', ')
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)) }

function fallbackCenter(index, count) {
  if (count <= 1) return { x: 0.5, y: 0.5 }
  return { x: (index + 1) / (count + 1), y: 0.5 }
}

export function randomSeed() { return Math.floor(Math.random() * 2147483647) }

/**
 * Normalize tool-provided character entries: trim captions, clamp or fall back
 * centers, drop empty captions, cap at MAX_CHARS.
 * @param raw - array of `{ caption, center? }`.
 * @returns normalized `[{ caption, center }]`.
 */
export function normalizeCharacters(raw) {
  const list = Array.isArray(raw) ? raw.slice(0, MAX_CHARS) : []
  return list.map((c, i) => {
    const caption = normalizeRelationTags(String((c && c.caption) || '')).trim()
    const cx = c && c.center && typeof c.center.x === 'number' ? c.center.x : null
    const cy = c && c.center && typeof c.center.y === 'number' ? c.center.y : null
    const center = (cx !== null && cy !== null)
      ? { x: clamp(cx, 0.05, 0.95), y: clamp(cy, 0.05, 0.95) }
      : fallbackCenter(i, list.length)
    return { caption, center }
  }).filter((c) => c.caption.length > 0)
}

/** Build the V4.5 request body, field-for-field as the reference client does. */
export function buildRequestBody(plan) {
  const characterCaptions = plan.characters.map((c) => ({
    char_caption: c.caption,
    centers: [{ x: c.center.x, y: c.center.y }],
  }))
  const v4Prompt = {
    caption: { base_caption: plan.baseCaption, char_captions: characterCaptions },
    use_coords: false,
    use_order: true,
  }
  const v4NegativePrompt = {
    caption: {
      base_caption: plan.negative,
      char_captions: plan.characters.map((c) => ({ char_caption: '', centers: [{ x: c.center.x, y: c.center.y }] })),
    },
    legacy_uc: false,
    use_coords: false,
    use_order: true,
  }
  return {
    input: plan.baseCaption,
    model: MODEL,
    action: 'generate',
    parameters: {
      params_version: 3,
      width: plan.width,
      height: plan.height,
      scale: SCALE,
      sampler: 'k_euler_ancestral',
      steps: STEPS,
      seed: plan.seed,
      extra_noise_seed: plan.seed,
      n_samples: plan.batchSize,
      ucPreset: 3,
      qualityToggle: false,
      negative_prompt: plan.negative,
      noise_schedule: 'karras',
      legacy: false,
      legacy_uc: false,
      use_coords: false,
      legacy_v3_extend: false,
      autoSmea: false,
      sm: false,
      sm_dyn: false,
      dynamic_thresholding: false,
      cfg_rescale: 0.0,
      skip_cfg_above_sigma: null,
      deliberate_euler_ancestral_bug: false,
      prefer_brownian: true,
      stream: 'msgpack',
      v4_prompt: v4Prompt,
      v4_negative_prompt: v4NegativePrompt,
    },
  }
}

/** Minimal MessagePack decoder (sufficient for NovelAI stream frames). */
function decodeMsgpack(bytes) {
  let pos = 0
  const te = new TextDecoder()
  function u8() { return bytes[pos++] }
  function readUint(n) { let v = 0; for (let i = 0; i < n; i++) v = v * 256 + u8(); return v }
  function readBytes(n) { const b = bytes.subarray(pos, pos + n); pos += n; return b }
  function readString(n) { return te.decode(readBytes(n)) }
  function readValue() {
    const b = u8()
    if (b <= 0x7f) return b
    if (b >= 0xe0) return b - 256
    if (b >= 0xa0 && b <= 0xbf) return readString(b & 0x1f)
    if (b >= 0x90 && b <= 0x9f) { const n = b & 0xf; const a = []; for (let i = 0; i < n; i++) a.push(readValue()); return a }
    if (b >= 0x80 && b <= 0x8f) { const n = b & 0xf; const o = {}; for (let i = 0; i < n; i++) { const k = readValue(); o[String(k)] = readValue() } return o }
    switch (b) {
      case 0xc0: return null
      case 0xc2: return false
      case 0xc3: return true
      case 0xcc: return readUint(1)
      case 0xcd: return readUint(2)
      case 0xce: return readUint(4)
      case 0xcf: return readUint(8)
      case 0xd0: { const v = readUint(1); return v >= 128 ? v - 256 : v }
      case 0xd1: { const v = readUint(2); return v >= 32768 ? v - 65536 : v }
      case 0xd2: { const v = readUint(4); return v >= 2147483648 ? v - 4294967296 : v }
      case 0xd3: return readUint(8)
      case 0xd9: return readString(readUint(1))
      case 0xda: return readString(readUint(2))
      case 0xdb: return readString(readUint(4))
      case 0xc4: return readBytes(readUint(1))
      case 0xc5: return readBytes(readUint(2))
      case 0xc6: return readBytes(readUint(4))
      case 0xdc: { const n = readUint(2); const a = []; for (let i = 0; i < n; i++) a.push(readValue()); return a }
      case 0xdd: { const n = readUint(4); const a = []; for (let i = 0; i < n; i++) a.push(readValue()); return a }
      case 0xde: { const n = readUint(2); const o = {}; for (let i = 0; i < n; i++) { const k = readValue(); o[String(k)] = readValue() } return o }
      case 0xdf: { const n = readUint(4); const o = {}; for (let i = 0; i < n; i++) { const k = readValue(); o[String(k)] = readValue() } return o }
      default: throw new Error('unsupported msgpack code 0x' + b.toString(16))
    }
  }
  return readValue()
}

function parseFrames(bytes) {
  const frames = []
  let offset = 0
  while (offset + 4 <= bytes.length) {
    const size = ((bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3]) >>> 0
    if (size < 1 || size > 64 * 1024 * 1024) throw new Error('NovelAI 流帧大小无效: ' + size)
    if (offset + 4 + size > bytes.length) break
    frames.push(bytes.subarray(offset + 4, offset + 4 + size))
    offset += 4 + size
  }
  return frames
}

/**
 * Parse a raw msgpack stream body into final PNGs.
 * @param bytes - the full response body.
 * @returns `{ finals: Uint8Array[], intermediates: number }`.
 */
export function parseStream(bytes) {
  const finals = []
  let intermediates = 0
  let errorMsg = null
  for (const frame of parseFrames(bytes)) {
    let obj
    try { obj = decodeMsgpack(frame) } catch { continue }
    if (!obj || typeof obj !== 'object') continue
    const ev = obj.event_type
    if (ev === 'final') {
      const img = obj.image
      if (img && img.length) finals.push(img)
    } else if (ev === 'intermediate') {
      intermediates++
    } else if (ev === 'error') {
      errorMsg = String(obj.message || '未知错误')
    }
  }
  if (errorMsg) throw new Error('NovelAI 服务端报错: ' + errorMsg)
  if (finals.length === 0) throw new Error('NovelAI 未返回最终图片')
  return { finals, intermediates }
}

/**
 * Run one NovelAI V4.5 Full generation through curl (optionally proxied).
 * @param options - baseCaption, characters (normalized), negative, sizePreset,
 *   seed, batchSize, token, proxy, signal.
 * @returns `{ finals, intermediates }`.
 */
export async function generateImage(options) {
  const {
    baseCaption,
    characters = [],
    negative,
    sizePreset = 'PORTRAIT',
    seed,
    batchSize = 1,
    token,
    proxy = DEFAULT_PROXY,
    signal,
  } = options
  if (!token) throw new Error('未配置 NovelAI Token（config.token / 环境变量 NOVELAI_API_KEY / 工具参数 token）')
  const preset = SIZES[normalizeSizePreset(sizePreset)]
  const body = buildRequestBody({
    baseCaption,
    characters,
    negative,
    width: preset.width,
    height: preset.height,
    seed,
    batchSize: Math.min(MAX_BATCH, Math.max(1, batchSize)),
  })

  const dir = mkdtempSync(join(tmpdir(), 'nai-'))
  const bodyPath = join(dir, 'body.json')
  const outPath = join(dir, 'out.bin')
  try {
    writeFileSync(bodyPath, JSON.stringify(body))

    const args = [
      'curl.exe', '-sS', '-X', 'POST', ENDPOINT,
      '-H', 'Authorization: Bearer ' + token,
      '-H', 'Content-Type: application/json',
      '-H', 'Accept: application/octet-stream',
      '--data-binary', '@' + bodyPath,
      '-o', outPath,
      '-w', '%{http_code}',
      '--connect-timeout', '30',
      '--max-time', '540',
    ]
    if (/^https?:\/\//.test(String(proxy || '').trim())) args.push('-x', String(proxy).trim())

    let stdout = ''
    let stderr = ''
    try {
      const result = await execFileAsync(args[0], args.slice(1), {
        encoding: 'utf8',
        maxBuffer: 1024 * 1024,
        timeout: 600000,
        signal,
      })
      stdout = (result.stdout || '').trim()
      stderr = (result.stderr || '').trim()
    } catch (e) {
      stdout = (e.stdout || '').toString().trim()
      stderr = String(e.stderr || e.message || '')
      if (e.killed || e.signal) throw new Error('NovelAI 请求已取消或超时')
      if (!stdout) throw new Error('curl 请求失败: ' + (stderr || e.message))
    }

    if (!/^2/.test(stdout)) {
      let errBody = ''
      try { errBody = readFileSync(outPath, 'utf8').slice(0, 1000) } catch { /* no body */ }
      throw new Error('NovelAI HTTP ' + stdout + (errBody ? ': ' + errBody : ''))
    }

    const buf = readFileSync(outPath)
    return parseStream(new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength))
  } finally {
    try { rmSync(dir, { recursive: true, force: true }) } catch { /* best-effort cleanup */ }
  }
}

/**
 * Write final PNGs to disk. Returns the saved file paths.
 * @param finals - PNG byte arrays.
 * @param seed - the generation seed (used in filenames).
 * @param outDir - destination directory (created recursively).
 * @returns absolute file paths.
 */
export function savePngs(finals, seed, outDir) {
  mkdirSync(outDir, { recursive: true })
  const paths = []
  for (let i = 0; i < finals.length; i++) {
    const p = join(outDir, `nai-${seed}-${i + 1}.png`)
    writeFileSync(p, finals[i])
    paths.push(p)
  }
  return paths
}

/**
 * Query the Danbooru tag-suggest API (Chinese / English / pinyin searchable,
 * no auth). Used to look up the exact official tags before designing a prompt.
 * @param query - the search keyword.
 * @param limit - max results to return (default 8).
 * @param signal - optional AbortSignal.
 * @returns `[{ name, cnName, count, category }]`.
 */
export async function suggestTags(query, limit = 8, signal) {
  const q = String(query || '').trim()
  if (!q) return []
  const url = 'https://tagsuggest.zeabur.app/api/tags/suggest?q=' + encodeURIComponent(q)
  const timeoutSignal = AbortSignal.timeout(10000)
  const combined = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal
  const res = await fetch(url, { signal: combined, headers: { Accept: 'application/json' } })
  if (!res.ok) throw new Error('Tag 搜索 API HTTP ' + res.status)
  const data = await res.json()
  const list = Array.isArray(data && data.results) ? data.results : []
  return list.slice(0, Math.max(1, limit)).map((r) => ({
    name: String(r.name || ''),
    cnName: String(r.cn_name || ''),
    count: typeof r.count === 'number' ? r.count : 0,
    category: typeof r.category === 'number' ? r.category : 0,
  }))
}
