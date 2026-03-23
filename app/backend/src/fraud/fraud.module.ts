import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { FraudService } from './fraud.service';
import { PrismaModule } from '../prisma/prisma.module';

@Module({
  imports: [HttpModule, PrismaModule],
  providers: [FraudService],
  exports: [FraudService],
})
export class FraudModule {}
