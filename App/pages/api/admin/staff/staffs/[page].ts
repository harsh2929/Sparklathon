import PostgresClient from '@lib/database';
import { staffQueries } from '@lib/sql';
import { Category } from '@ts-types/generated';
import type { NextApiRequest, NextApiResponse } from 'next';

class Handler extends PostgresClient {
  constructor() {
    super();
  }

  execute = async (req: NextApiRequest, res: NextApiResponse) => {
    const { query, method } = req;
    const page = parseInt(query.page as string, 10);
    const offset = page === 0 ? 0 : (page - 1) * this.limit;
    try {
      switch (method) {
        case this.GET: {
          const results = await this.tx(async (client) => {
            await this.authorization(client, req, res, true);
            const { rows: staffs } = await client.query<Category, number>(
              staffQueries.getStaffs(),
              [this.limit, offset]
            );
            const { rows } = await client.query<{ count: number }, any>(
              staffQueries.staffsCount(),
              []
            );
            const { count: value } = rows[0];
            const count = value ? Number(value) : 0;
            return { staffs, count };
          });
          return res.status(200).json(results);
        }
        default:
          res.setHeader('Allow', ['GET']);
          res.status(405).end(`There was some error!`);
      }
    } catch (error) {
      res.status(500).json({
        error: {
          type: this.ErrorNames.SERVER_ERROR,
          message: error?.message,
          from: 'staffs'
        }
      });
    }
  };
}

const { execute } = new Handler();

export default execute;
