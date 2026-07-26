import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob } from 'astro/loaders';

const lessons = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/lessons' }),
  schema: z.object({
    title: z.string(),
    subtitle: z.string(),
    pillar: z.enum(['fundamentos', 'ingesta', 'recuperacion', 'agentic', 'ragops']),
    pillarName: z.string(),
    lessonNum: z.number(),
    description: z.string(),
    keywords: z.string(),
    ogSection: z.string(),
    pubDate: z.string(),
    quizzes: z.array(z.object({
      id: z.string(),
      question: z.string(),
      options: z.array(z.object({
        text: z.string(),
        correct: z.boolean()
      }))
    }))
  })
});

export const collections = { lessons };
