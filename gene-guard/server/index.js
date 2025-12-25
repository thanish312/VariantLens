// index.js
import 'dotenv/config'
import express from 'express'
import multer from 'multer'
import cors from 'cors'
import fs from 'fs'
import { GoogleGenAI } from '@google/genai'

/* -------------------- App Setup -------------------- */
const app = express()
app.use(cors())
app.use(express.json())

/* -------------------- Multer -------------------- */
const storage = multer.diskStorage({
  destination(req, file, cb) {
    const dir = 'uploads'
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true })
    cb(null, dir)
  },
  filename(req, file, cb) {
    cb(null, `${Date.now()}-${file.originalname.replace(/\s+/g, '_')}`)
  }
})

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter(req, file, cb) {
    if (
      file.mimetype === 'text/plain' ||
      file.originalname.endsWith('.vcf') ||
      file.originalname.endsWith('.txt')
    ) {
      cb(null, true)
    } else {
      cb(new Error('Only .vcf or .txt files allowed'), false)
    }
  }
})

/* -------------------- Gemini Init -------------------- */
if (!process.env.GEMINI_API_KEY) {
  console.error('GEMINI_API_KEY missing')
  process.exit(1)
}

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
})

/* -------------------- Variant Rules -------------------- */
const MIN_VARIANT_QUALITY = 30
const MAX_VARIANTS_TO_REPORT = 5
const MAX_ALLELE_FREQUENCY_COMMON = 0.01

const INTERESTING_CONSEQUENCES = [
  'transcript_ablation',
  'splice_acceptor_variant',
  'splice_donor_variant',
  'stop_gained',
  'frameshift_variant',
  'stop_lost',
  'start_lost',
  'inframe_insertion',
  'inframe_deletion',
  'missense_variant',
  'protein_altering_variant',
  'splice_region_variant',
  'synonymous_variant'
]

const CONSEQUENCE_PRIORITY = Object.fromEntries(
  INTERESTING_CONSEQUENCES.map((c, i) => [c, i])
)

/* -------------------- Annotation Mapper -------------------- */
function mapAnnotation(raw, format, source) {
  const fields = raw.split('|')
  const data = Object.fromEntries(format.map((k, i) => [k, fields[i]]))

  if (source === 'snpeff') {
    return {
      gene: data.GENE || data.GENE_NAME,
      consequence: data.ANNOTATION || data.EFFECT,
      impact: data.IMPACT
    }
  }

  if (source === 'vep') {
    return {
      gene: data.SYMBOL,
      consequence: data.Consequence?.split('&')[0],
      impact: data.IMPACT
    }
  }

  if (source === 'bcftools') {
    return {
      gene: data.gene,
      consequence: data.consequence,
      impact: data.impact
    }
  }

  return {}
}

/* -------------------- VCF Parser -------------------- */
function extractMeaningfulGeneInfo(fileContent) {
  if (!fileContent?.trim()) {
    return { error: 'Empty VCF', variants: [], genes: [] }
  }

  const lines = fileContent.split('\n')

  let annFormat = null
  let csqFormat = null
  let bcsqFormat = null

  for (const line of lines) {
    if (!line.startsWith('##')) continue

    if (line.startsWith('##INFO=<ID=ANN')) {
      annFormat = line.match(/Format: (.+?)">/)?.[1]?.split('|')
    }
    if (line.startsWith('##INFO=<ID=CSQ')) {
      csqFormat = line.match(/Format: (.+?)">/)?.[1]?.split('|')
    }
    if (line.startsWith('##INFO=<ID=BCSQ')) {
      bcsqFormat = line.match(/Format: (.+?)">/)?.[1]?.split('|')
    }
  }

  const dataLines = lines.filter(l => l && !l.startsWith('#'))
  const variants = []

  for (const line of dataLines) {
    if (variants.length >= MAX_VARIANTS_TO_REPORT) break

    const cols = line.split('\t')
    if (cols.length < 8) continue

    const [chrom, pos, , ref, altStr, qualStr, filter, infoStr] = cols
    const qual = parseFloat(qualStr)

    if (isNaN(qual) || qual < MIN_VARIANT_QUALITY) continue
    if (filter !== 'PASS' && filter !== '.') continue

    const info = {}
    infoStr.split(';').forEach(f => {
      const [k, v] = f.split('=')
      info[k] = v ?? true
    })

    let maxAf = null
    if (info.AF) {
      maxAf = Math.max(...info.AF.split(',').map(Number).filter(n => !isNaN(n)))
    } else if (info.AC && info.AN) {
      const ac = info.AC.split(',').map(Number)
      const an = Number(info.AN)
      if (an > 0) maxAf = Math.max(...ac.map(a => a / an))
    }

    if (maxAf !== null && maxAf > MAX_ALLELE_FREQUENCY_COMMON) continue

    let annotations = []

    if (info.ANN && annFormat) {
      annotations = info.ANN.split(',').map(a =>
        mapAnnotation(a, annFormat, 'snpeff')
      )
    } else if (info.CSQ && csqFormat) {
      annotations = info.CSQ.split(',').map(a =>
        mapAnnotation(a, csqFormat, 'vep')
      )
    } else if (info.BCSQ && bcsqFormat) {
      annotations = info.BCSQ.split(',').map(a =>
        mapAnnotation(a, bcsqFormat, 'bcftools')
      )
    }

    const best = annotations
      .filter(a => INTERESTING_CONSEQUENCES.includes(a.consequence))
      .sort(
        (a, b) =>
          CONSEQUENCE_PRIORITY[a.consequence] -
          CONSEQUENCE_PRIORITY[b.consequence]
      )[0]

    if (!best) continue

    variants.push({
      representation: `${chrom}:${pos} ${ref}>${altStr.split(',')[0]}`,
      gene: best.gene || 'N/A',
      consequence: best.consequence,
      impact: best.impact || 'N/A',
      qual
    })
  }

  if (!variants.length) {
    return { error: 'No variants passed filters', variants: [], genes: [] }
  }

  const genes = [...new Set(variants.map(v => v.gene).filter(g => g !== 'N/A'))]
  const variantSummary = variants
    .map(v => `${v.representation} (Gene: ${v.gene}, Effect: ${v.consequence})`)
    .join('; ')

  return { variants, genes, variantSummary, error: null }
}

/* -------------------- Prompt -------------------- */
const promptTemplate = process.env.GENAI_PROMPT
  ? process.env.GENAI_PROMPT
  : fs.readFileSync('prompt.txt', 'utf8')

/* -------------------- API -------------------- */
app.post('/api/predict', upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' })

  const filePath = req.file.path

  try {
    const fileContent = fs.readFileSync(filePath, 'utf8')
    const { variants, genes, variantSummary, error } =
      extractMeaningfulGeneInfo(fileContent)

    if (error) return res.status(400).json({ error })

    const prompt = promptTemplate
      .replace(/\$\{age\}/g, '25')
      .replace(/\$\{gender\}/g, 'female')
      .replace(/\$\{variantSummary\}/g, variantSummary)
      .replace(/\$\{geneList\}/g, genes.join(', '))
      .replace(
        /\$\{detailedVariantInfo\}/g,
        variants
          .map(
            v =>
              `- ${v.representation}, Gene: ${v.gene}, Effect: ${v.consequence}`
          )
          .join('\n')
      )

    const result = await ai.models.generateContent({
      model: 'gemini-1.5-flash',
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.5, maxOutputTokens: 3000 }
    })

    const rawText =
      typeof result.text === 'function'
        ? await result.text()
        : result.candidates?.[0]?.content?.parts?.[0]?.text

    const jsonMatch = rawText.match(/```json\s*([\s\S]*?)```|(\{[\s\S]*\})/)
    const parsed = JSON.parse(jsonMatch?.[1] || jsonMatch?.[2] || rawText)

    res.json({
      aiAnalysis: parsed,
      processedVariantsInput: variants,
      identifiedGenesForPrompt: genes
    })
  } catch (e) {
    res.status(500).json({ error: e.message })
  } finally {
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath)
  }
})

/* -------------------- Server -------------------- */
const PORT = process.env.PORT || 5174
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`)
})
