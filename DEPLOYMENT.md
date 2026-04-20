# Deployment Instructions for Netfix Project on Vercel

The project is now configured for Vercel. Follow these steps to put it live:

### Option A: Using Vercel CLI (Recommended for fast testing)
1. Open your terminal in the project folder.
2. Run `vercel` and follow the prompts (log in if needed).
3. Once the preview is ready, run `vercel --prod` to deploy to production.

### Option B: Using GitHub (Recommended for automatic updates)
1. Push your code to a GitHub repository (including the `model` folder).
2. Go to [vercel.com](https://vercel.com) and click **"Add New Project"**.
3. Import your GitHub repository.
4. Vercel will automatically detect the settings and deploy.

### Important Notes:
- **Model Files**: Ensure `model/model.pkl` and `model/cleaned_data.csv` are uploaded. Vercel will use them to serve predictions.
- **Dependencies**: Vercel will automatically install everything in `requirements.txt`.
- **D: Drive**: The code will ignore the `D:/DATASET` path on Vercel and correctly fall back to the local `model/` files.
